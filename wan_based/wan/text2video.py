# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import logging
import math
import os
import random
import sys
import types
from contextlib import contextmanager
from functools import partial

import torch
import torch.cuda.amp as amp
import torch.distributed as dist
from tqdm import tqdm
import random
import cv2

from .distributed.fsdp import shard_model
from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae import WanVAE
from .utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                               get_sampling_sigmas, retrieve_timesteps)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from .utils.utils import cache_video, cache_image, str2bool
from thop import profile
from thop.utils import clever_format

from torch.nn import functional

class WanT2V:

    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
    ):
        r"""
        Initializes the Wan text-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_usp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of USP.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None)

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        logging.info(f"Creating WanModel from {checkpoint_dir}")
        self.model = WanModel.from_pretrained(checkpoint_dir)
        self.model.eval().requires_grad_(False)

        if use_usp:
            from xfuser.core.distributed import \
                get_sequence_parallel_world_size

            from .distributed.xdit_context_parallel import (usp_attn_forward,
                                                            usp_dit_forward)
            for block in self.model.blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn)
            self.model.forward = types.MethodType(usp_dit_forward, self.model)
            self.sp_size = get_sequence_parallel_world_size()
        else:
            self.sp_size = 1

        if dist.is_initialized():
            dist.barrier()
        if dit_fsdp:
            self.model = shard_fn(self.model)
        else:
            self.model.to(self.device)

        self.sample_neg_prompt = config.sample_neg_prompt

        #### involve memory
        self.temporal_memory_len = 4
        self.temporal_memory_stamp = [(i+1)*2 for i  in range(self.temporal_memory_len)]  # time stamp
        self.temporal_memory_confidence_infor = [0]*self.temporal_memory_len
        self.temporal_memory_step_infor = [0] * self.temporal_memory_len
        self.memory_bank = []


    def generate(self,
                 input_prompt,
                 long_generate_setting,
                 size=(1280, 720),
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True):
        r"""
        Generates video frames from text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation
            size (tupele[`int`], *optional*, defaults to (1280,720)):
                Controls video resolution, (width,height).
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed.
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from size)
                - W: Frame width from size)
        """
        exp_name = long_generate_setting["exp_name"]               # _new_propmt
        save_dir = f"{long_generate_setting['save_dir']}/{exp_name}"
        os.makedirs(save_dir,exist_ok=True)
        step_video_save_dir_name = long_generate_setting["step_video_save_dir_name"]
        save_dir_step_video = os.path.join(save_dir,step_video_save_dir_name)   # step_video
        os.makedirs(save_dir_step_video,exist_ok=True)


        # preprocess
        F = frame_num       # 81
        target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
                        size[1] // self.vae_stride[1],
                        size[0] // self.vae_stride[2])                              # (16, 21, 60, 104)  d, t, h, w

        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size          # 32760

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)


        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
                del self.text_encoder
                # 使用垃圾回收来清理没有用到的对象
                import gc
                gc.collect()
                
                # 清理 CUDA 缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        noise = [
            torch.randn(
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=self.device,
                generator=seed_g)
        ]                                                   # ([16, 21, 60, 104])

        
        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # evaluation mode
        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():

            if sample_solver == 'unipc':            # by this
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,       # 1000
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)    # sampling_steps: 50
                timesteps = sample_scheduler.timesteps                  # 50个值： 999，,995，..., 92
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            latents = noise

            arg_c = {'context': context, 'seq_len': seq_len}
            arg_null = {'context': context_null, 'seq_len': seq_len}

            for _, t in enumerate(tqdm(timesteps)):
                latent_model_input = latents
                timestep = [t]

                timestep = torch.stack(timestep)

                self.model.to(self.device)
                noise_pred_cond = self.model(
                    latent_model_input, t=timestep, **arg_c)[0]
                noise_pred_uncond = self.model(  
                    latent_model_input, t=timestep, **arg_null)[0]

                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latents = [temp_x0.squeeze(0)]


            save_file = os.path.join(save_dir_step_video,"saved_video_initial.mp4")  
            self.decode_and_save_video(latents, save_file)

            save_path = os.path.join(save_dir_step_video,"latent_initial.pt") 
            torch.save(latents, save_path)

            x0 = latents
            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()
            if self.rank == 0:
                videos = self.vae.decode(x0)

        del noise, latents
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None



    def initial_quene_memory_dict(self,seq_len):
        quene_memory_dict = {}
        quene_memory_dict["prev_sample"]= []
        quene_memory_dict["model_outputs"]= []
        quene_memory_dict["timestep_list"]= []
        quene_memory_dict["last_sample"]= []
        quene_memory_dict["this_order"]= []

        for _ in range( seq_len ):
            quene_memory_dict["prev_sample"].append(None)
            quene_memory_dict["model_outputs"].append([None,None])
            quene_memory_dict["timestep_list"].append([None,None])
            quene_memory_dict["last_sample"].append(None)
            quene_memory_dict["this_order"].append(None)

        return quene_memory_dict

    def decode_and_save_video(self, latents_initial,save_file):
        saved_latents_1 = latents_initial                 
        saved_latents_1 = torch.cat(saved_latents_1,dim=1)     #[0] , ([16, 21, 60, 104])
        videos_1 = self.vae.decode([saved_latents_1])[0]       # [0] ([3, 81, 480, 832])
        
        cache_video(
            tensor=videos_1[None],
            save_file=save_file,
            fps=16,
            nrow=1,
            normalize=True,
            value_range=(-1, 1))

    def adjust_judge(self,saved_latents,new_gen_latents,sim_result,threshold=0.002): 
            adjust_flag = False
            feats = saved_latents
            feats = torch.stack(feats,dim=0)
            f,d,_,h,w = feats.shape
            feats = feats.view(f,d,-1) # f,16,6240 -> f,6240,16
            frame_vecs = feats.mean(dim=1)  
            frame_vecs_1 = functional.normalize(frame_vecs, p=2, dim=1)

            feats = new_gen_latents
            feats = torch.stack(feats,dim=0)
            f,d,_,h,w = feats.shape
            feats = feats.view(f,d,-1) # f,16,6240 -> f,6240,16
            frame_vecs = feats.mean(dim=1)  
            frame_vecs_2 = functional.normalize(frame_vecs, p=2, dim=1)

            sim_matrix = frame_vecs_1 @ frame_vecs_2.T
            sim_matrix_weight = sim_matrix.mean(0).mean(0)

            if len(sim_result) > 0:
                latest_sim = sim_result[-1]
                sim_result.append(sim_matrix_weight) 
                if latest_sim - sim_matrix_weight> threshold:
                    ##### begin adjust
                    adjust_flag = True
                    print(f"begin adjusting, sim_result: {sim_result}")
                    
            else:
                sim_result.append(sim_matrix_weight) 
            return adjust_flag, sim_result   

    def generate_by_miga(self,
                 input_prompt,
                 size=(1280, 720),
                 frame_num_initial_wan=81,      ## 4n+1. 
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True,
                 long_generate_setting=None):

        """
          MIGA
        """
        exp_name = long_generate_setting.exp_name 
        save_dir = f"{long_generate_setting.save_dir}/{exp_name}"
        os.makedirs(save_dir,exist_ok=True)
        saw_width =long_generate_setting.saw_width           # for stage1 
        
        ############################################ initial preprocess ############################################
        F = frame_num_initial_wan       # 81
        target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
                        size[1] // self.vae_stride[1],
                        size[0] // self.vae_stride[2])                              # (16, 21, 60, 104)  d, t, h, w
        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size          # 32760

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
                del self.text_encoder
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]


        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)
        self.model.to(self.device)
        self.model.eval()

        # evaluation mode
        sampling_steps = long_generate_setting.sampling_steps
        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():

            if sample_solver == 'unipc':            # by this
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,       # 1000
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)    # sampling_steps: 50
                timesteps = sample_scheduler.timesteps                  # 50个值： 999，,995，..., 92
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            ############################################ initial  sample  for initialize the quene   ############################################ 
            arg_c = {'context': context, 'seq_len': seq_len}
            arg_null = {'context': context_null, 'seq_len': seq_len}

            noise = [
                            torch.randn(
                                target_shape[0],
                                target_shape[1],
                                target_shape[2],
                                target_shape[3],
                                dtype=torch.float32,
                                device=self.device,
                                generator=seed_g)  
                        ]                                                   # ([16, 21, 60, 104])
            # sample videos
            latents = noise

            print(f"initial  sample  for initialize the quene ...")
            for _, t in enumerate(tqdm(timesteps)):
                latent_model_input = latents
                timestep = [t]

                timestep = torch.stack(timestep)

                self.model.to(self.device)
                noise_pred_cond = self.model(
                    latent_model_input, t=timestep, **arg_c)[0]
                noise_pred_uncond = self.model(
                    latent_model_input, t=timestep, **arg_null)[0]

                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latents = [temp_x0.squeeze(0)]

            latents_initial = latents

            if long_generate_setting.save_inter_results:
                save_file = os.path.join(save_dir,"saved_video_initial.mp4")  
                self.decode_and_save_video(latents, save_file)


            ############################################  initialize the quene   ############################################ 
            print(f"initialize the quene ...")
            long_latent_len = long_generate_setting.sampling_steps *saw_width        # the number of latent frames
            initial_temporal_len = latents_initial[0].shape[1]
            latent_list,timesteps_list, steps_list = [],[],[]

            noise = [ torch.randn(
                                target_shape[0],
                                long_latent_len, 
                                target_shape[2],
                                target_shape[3],
                                dtype=torch.float32,
                                device=self.device,
                                generator=seed_g)      
                        ]           

            ############  process_0: Initialize the last initial_temporal_len latents using the existing initial_temporal_len latents. 
            noise_index = int((long_latent_len - initial_temporal_len)/saw_width) 
            frame_index = 0
            
            for chunk_index in range(int(initial_temporal_len/saw_width)):
                for chunk_index_inner in range(saw_width):
                    latents_item  = latents_initial[0][:,chunk_index*saw_width+chunk_index_inner].unsqueeze(1)
                    noise_item = torch.randn_like(latents_item) 
                    timesteps_list.append(timesteps[ sampling_steps - noise_index]  )
                    steps_list.append(   sampling_steps - noise_index )
                    latents_item_with_noise0 = sample_scheduler.add_noise(latents_item, noise_item,timesteps[sampling_steps -noise_index].unsqueeze(0))
                    latent_list.append(latents_item_with_noise0)
                noise_index += 1
            
            #############  process_1: Progressively complete the initialization of the remaining latent frames.
            ### involve memory 
            initial_frame_num = target_shape[1]
            self.temporal_memory_len = long_generate_setting.temporal_memory_len
            quene_memory_dict = self.initial_quene_memory_dict(len(latent_list) + saw_width)
            initial_frame_num_adjust = initial_frame_num - self.temporal_memory_len

            initial_iter_nums = long_generate_setting.long_iter_nums
            if initial_iter_nums > (long_latent_len - initial_frame_num)/saw_width:
                initial_iter_nums = int((long_latent_len - initial_frame_num)/saw_width)
            
            ### begin of Stage 2 ( Denoising at a Unified Noise Level)
            fifo_end_noise_index = long_generate_setting.sampling_steps - 10
            noise_index = long_generate_setting.sampling_steps  
            
            ### resample  infor
            sim_result = []
            adjust_num = 0
            adjust_adopt_num = 0

            for timesteps_iter_index in tqdm(range( initial_iter_nums )):
                for chunk_index_inner in range(saw_width):
                    latents_item  = latents_initial[0][:,-1].unsqueeze(1)
                    noise_item = torch.randn_like(latents_item)
                    timesteps_list.append(timesteps[ sampling_steps - noise_index]  )
                    steps_list.append(   sampling_steps - noise_index )
                    latents_item_with_noise0 = sample_scheduler.add_noise(latents_item, noise_item,timesteps[sampling_steps -noise_index].unsqueeze(0))
                    latent_list.append(latents_item_with_noise0)

                if len(latent_list) == long_latent_len:
                    break

                if steps_list[0] >= fifo_end_noise_index-2:
                    break

                ####  determine the chunk infor
                frame_len = len(latent_list)
                slide_size = 10

                chunk_num = int((frame_len-initial_frame_num_adjust)/slide_size) + 2
                remainder = (frame_len-initial_frame_num_adjust)%slide_size

                for chunk_index in range(chunk_num):
                    if chunk_index == 0:
                        start_index = 0
                        cal_len = remainder 
                        if cal_len == 0:
                            continue
                    elif chunk_index == chunk_num-1:
                        cal_len = initial_frame_num_adjust
                        start_index = frame_len - initial_frame_num_adjust
                    elif chunk_index > 0 and chunk_index < chunk_num-1:
                        start_index = remainder + (chunk_index-1)*slide_size
                        cal_len = slide_size
                    else:
                        print("wrong in initial !!!!!!!!")
                    end_index = start_index + initial_frame_num_adjust

                    latent_model_input = latent_list[start_index:end_index].copy() 
                    t_item = timesteps_list[start_index:end_index].copy()
                    
                    #### involve memory
                    if start_index < self.temporal_memory_len:
                        ## Retrieve latents from later
                        latent_model_input = latent_list[start_index:end_index+self.temporal_memory_len].copy()
                        t_item = timesteps_list[start_index:end_index+self.temporal_memory_len].copy()
                        index_bais = 0
                    else:
                        index_bais = self.temporal_memory_len
                        if start_index >= self.temporal_memory_len*saw_width:
                            memory_latents_list = []
                            memory_steps_list = []
                            for memory_index in reversed(range(self.temporal_memory_len)):
                                memory_latents_list.append(latent_list[start_index-(3 + (memory_index)*saw_width)])
                                memory_steps_list.append(timesteps_list[start_index-(3 + (memory_index)*saw_width)])
                                if start_index-(3 + (memory_index)*saw_width) < 0 or (3 + (memory_index)*saw_width) <0:
                                    print(f"!!!!!!!!!wrong memory_index,{start_index},{memory_index},{saw_width}")
                            latent_model_input = memory_latents_list+ latent_model_input
                            t_item = memory_steps_list + t_item

                        else:
                            memory_sample_slide = int(start_index/self.temporal_memory_len)
                            memory_latents_list = []
                            memory_steps_list = []
                            for memory_index in reversed(range(self.temporal_memory_len)):
                                memory_latents_list.append(latent_list[start_index- (memory_index+1)*memory_sample_slide])
                                memory_steps_list.append(timesteps_list[start_index-(memory_index+1)*memory_sample_slide])
                                if start_index- (memory_index+1)*memory_sample_slide < 0 or (memory_index+1)*memory_sample_slide < 0:
                                    print(f"!!!!!!!!!wrong memory_index,{start_index},{memory_index},{memory_sample_slide}")
                            latent_model_input = memory_latents_list+ latent_model_input
                            t_item = memory_steps_list + t_item

                    latent_model_input = [torch.cat(latent_model_input, dim=1)]
                    t_item = torch.stack(t_item)
                    timestep = t_item    

                    #### predict noise by model
                    noise_pred_cond = self.model(
                        latent_model_input, t=timestep, **arg_c)[0]
                    noise_pred_uncond = self.model(
                        latent_model_input, t=timestep, **arg_null)[0]
                    noise_pred = noise_pred_uncond + guide_scale * (
                        noise_pred_cond - noise_pred_uncond)       

                    ### sample_by_frame_wise 
                    for frame_index in range(cal_len):
                        load_index = start_index + frame_index
                        
                        given_dict = {}
                        given_dict["model_outputs"]= quene_memory_dict["model_outputs"][load_index].copy()
                        given_dict["timestep_list"]= quene_memory_dict["timestep_list"][load_index].copy()
                        given_dict["last_sample"]= quene_memory_dict["last_sample"][load_index]
                        given_dict["this_order"]= quene_memory_dict["this_order"][load_index]
                        
                        temp_x0_output_dict = sample_scheduler.step_by_different_e(
                            noise_pred[:,frame_index + index_bais ].unsqueeze(0).unsqueeze(2),    
                            timestep[frame_index  + index_bais ],
                            latent_model_input[0][:,frame_index + index_bais ].unsqueeze(0).unsqueeze(2),  
                            given_dict=given_dict,
                            return_dict=False,
                            generator=seed_g)
                        
                        ### save & update 
                        for key_item in  temp_x0_output_dict:
                            if isinstance(temp_x0_output_dict[key_item], list):
                                quene_memory_dict[key_item][load_index] = temp_x0_output_dict[key_item].copy()
                            else:
                                quene_memory_dict[key_item][load_index] = temp_x0_output_dict[key_item]

                        latent_list[load_index] = quene_memory_dict["prev_sample"][load_index].squeeze(0)
                        steps_list[load_index] = steps_list[load_index] + 1
                        timesteps_list[load_index] = timesteps[  steps_list[load_index] ]
                    
                ############# resample process (reflection and correction processing)
                judged_step = 10 
                if steps_list[0] > judged_step +2 and long_generate_setting.involve_resample:
                    judged_index = int(len(steps_list)/saw_width - judged_step)
                    judged_latents = latent_list[judged_index*saw_width:(judged_index+1)*saw_width]
                    judged_steps = steps_list[judged_index*saw_width:(judged_index+1)*saw_width]

                    early_latents = latent_list[(judged_index-2)*saw_width:judged_index*saw_width]
                    adjust_flag,sim_result =self.adjust_judge(judged_latents,early_latents,sim_result,threshold=long_generate_setting.resample_threshold)
                    print(f"%%%%%%%%%%%%% sim_result_latents:{sim_result}  at steps_list[0]: {steps_list[0]}")

                    if adjust_flag == True:
                        adjust_num+=1
                        guide_latents = early_latents.copy()
                        guide_steps = steps_list[(judged_index-2)*saw_width:(judged_index)*saw_width]
                        guide_timesteps = timesteps_list[(judged_index-2)*saw_width:(judged_index)*saw_width]

                        ### Progressively guide generation
                        adjust_sample_res = []
                        adjust_quene_memory_dict_res = []
                        for adjust_sample_index in range(3):
                            adjust_latents_list = guide_latents.copy()
                            adjust_steps_list = guide_steps.copy()
                            adjust_timesteps_list = guide_timesteps.copy()
                            adjust_quene_memory_dict = self.initial_quene_memory_dict(len(adjust_latents_list))

                            for adjust_timesteps_iter_index in tqdm(range( judged_step)):
                                for chunk_index_inner in range(saw_width):
                                    latents_item  = latents_initial[0][:,-1].unsqueeze(1)
                                    noise_item = torch.randn_like(latents_item) 
                                    adjust_timesteps_list.append(timesteps[ 0]  )
                                    adjust_steps_list.append( 0 )

                                    latents_item_with_noise0 = sample_scheduler.add_noise(latents_item, noise_item,timesteps[sampling_steps -noise_index].unsqueeze(0))
                                    adjust_latents_list.append(latents_item_with_noise0)

                                for chunk_index_inner in range(saw_width):
                                    adjust_quene_memory_dict["prev_sample"].append(None)
                                    adjust_quene_memory_dict["model_outputs"].append([None,None])
                                    adjust_quene_memory_dict["timestep_list"].append([None,None])
                                    adjust_quene_memory_dict["last_sample"].append(None)
                                    adjust_quene_memory_dict["this_order"].append(None)

                                # if len(adjust_latents_list) == (judged_step +2 +1)*saw_width: 
                                #     break
                                ####  determine the chunk infor
                                adjust_frame_len = len(adjust_latents_list)
                                slide_size = 10

                                chunk_num = int((adjust_frame_len-initial_frame_num)/slide_size) + 2
                                remainder = (adjust_frame_len-initial_frame_num)%slide_size

                                for chunk_index in range(chunk_num):
                                    if chunk_index == 0:
                                        start_index = 0
                                        cal_len = remainder 
                                        if cal_len == 0:
                                            continue
                                    elif chunk_index == chunk_num-1:
                                        cal_len = initial_frame_num
                                        start_index = adjust_frame_len - initial_frame_num
                                    elif chunk_index > 0 and chunk_index < chunk_num-1:
                                        start_index = remainder + (chunk_index-1)*slide_size
                                        cal_len = slide_size
                                    else:
                                        print("wrong in initial stage2!!!!!!!!")
                                    end_index = start_index + initial_frame_num

                                    latent_model_input = adjust_latents_list[start_index:end_index].copy()    
                                    latent_model_input = [torch.cat(latent_model_input, dim=1)]   
                                    
                                    t_item = adjust_timesteps_list[start_index:end_index]
                                    t_item = torch.stack(t_item)
                                    timestep = t_item     

                                    noise_pred_cond = self.model(
                                        latent_model_input, t=timestep, **arg_c)[0]
                                    noise_pred_uncond = self.model(
                                        latent_model_input, t=timestep, **arg_null)[0]

                                    noise_pred = noise_pred_uncond + guide_scale * (
                                        noise_pred_cond - noise_pred_uncond)  

                                    ###  sample_by_frame_wise 
                                    for frame_index in range(cal_len):
                                        load_index = start_index + frame_index
                                        if load_index < 2*saw_width:
                                            continue
                                        
                                        given_dict = {}
                                        given_dict["model_outputs"]= adjust_quene_memory_dict["model_outputs"][load_index].copy()
                                        given_dict["timestep_list"]= adjust_quene_memory_dict["timestep_list"][load_index].copy()
                                        given_dict["last_sample"]= adjust_quene_memory_dict["last_sample"][load_index]
                                        given_dict["this_order"]= adjust_quene_memory_dict["this_order"][load_index]
                                        
                                        temp_x0_output_dict = sample_scheduler.step_by_different_e(
                                            noise_pred[:,frame_index ].unsqueeze(0).unsqueeze(2),   
                                            timestep[frame_index ],
                                            latent_model_input[0][:,frame_index ].unsqueeze(0).unsqueeze(2), 
                                            given_dict=given_dict,
                                            return_dict=False,
                                            generator=seed_g)
                                        
                                        ### save & update
                                        for key_item in  temp_x0_output_dict:
                                            if isinstance(temp_x0_output_dict[key_item], list):
                                                adjust_quene_memory_dict[key_item][load_index] = temp_x0_output_dict[key_item].copy()
                                            else:
                                                adjust_quene_memory_dict[key_item][load_index] = temp_x0_output_dict[key_item]

                                        adjust_latents_list[load_index] = adjust_quene_memory_dict["prev_sample"][load_index].squeeze(0)
                                        adjust_steps_list[load_index] = adjust_steps_list[load_index] + 1
                                        adjust_timesteps_list[load_index] = timesteps[  adjust_steps_list[load_index] ]
                                    
                                # print(f"adjust_steps_list: {adjust_steps_list}") 
                        
                            adjust_sample_res.append(adjust_latents_list)
                            adjust_quene_memory_dict_res.append(adjust_quene_memory_dict)
                        
                        ## final update
                        max_sim_res,max_sim_index = -10,0
                        for adjust_sample_index in range(len(adjust_sample_res)):
                            adjust_latents_list = adjust_sample_res[adjust_sample_index]
                            new_gen_latents = adjust_latents_list[2*saw_width: 3*saw_width]
                        
                            _, sim_result_1 = self.adjust_judge(guide_latents,new_gen_latents,[],threshold=long_generate_setting.resample_threshold )
                            if sim_result_1[-1] > max_sim_res:
                                max_sim_res = sim_result_1[-1]
                                max_sim_index = adjust_sample_index
                        print(f" >>>>>>>>>>>>>>>>>>>>>>>>>>>> saved sim_res:{sim_result}, sim_result_1: {sim_result_1},max_sim_res: {max_sim_res},max_sim_index:{max_sim_index}")
                        if max_sim_res  > sim_result[-1]:
                            adjust_adopt_num+=1
                            print(f"************************** adjust")
                            ###### save the final res
                            adjust_latents_list = adjust_sample_res[max_sim_index]
                            adjust_quene_memory_dict = adjust_quene_memory_dict_res[max_sim_index]

                            latent_list_0 = latent_list[:(judged_index)*saw_width].copy()
                            latent_list_1 = adjust_latents_list[(-1*judged_step)*saw_width:].copy()
                            latent_list = latent_list_0 + latent_list_1

                            ### adjust_quene_memory_dict
                            for key_item in  adjust_quene_memory_dict:
                                quene_memory_dict[key_item] = quene_memory_dict[key_item][:(judged_index)*saw_width].copy() + adjust_quene_memory_dict[key_item][(-1*judged_step)*saw_width:].copy()
                            sim_result[-1] = max_sim_res
    
                for chunk_index_inner in range(saw_width):
                    quene_memory_dict["prev_sample"].append(None)
                    quene_memory_dict["model_outputs"].append([None,None])
                    quene_memory_dict["timestep_list"].append([None,None])
                    quene_memory_dict["last_sample"].append(None)
                    quene_memory_dict["this_order"].append(None)

            
            if long_generate_setting.save_inter_results:
                save_path = os.path.join(save_dir,"saved_video_initial_noise_resample_info.pt") 
                resample_infor = {}
                resample_infor["sim_result"] = sim_result
                resample_infor["resample_ratio"] = [initial_iter_nums,adjust_num, adjust_adopt_num ]
                torch.save(resample_infor, save_path)

                save_file = os.path.join(save_dir,"saved_video_initial_noise.mp4")  
                self.decode_and_save_video(latent_list, save_file)


            ############################################ Stage 1: Zigzag Iterative Denoising. ############################################  
            print("Stage 1: Zigzag Iterative Denoising.")          
            slide_size = 10
            initial_frame_num = target_shape[1]
            for item in quene_memory_dict:
                quene_memory_dict[item] = quene_memory_dict[item][:len(latent_list)]

            #### pop to save for Stage 2
            begin_pop_flag = False
            pop_quene_memory_dict = {} 
            saved_latents = []
            timesteps_iter_index = -1
            while True: 
                timesteps_iter_index +=1
                print(f"> timesteps_iter_index: {timesteps_iter_index} ")
                if len(saved_latents) >= long_generate_setting.long_iter_nums*saw_width:
                    break
                frame_len = len(latent_list)
                #### To avoid jitter caused by a fixed sliding-window trajectory, we design a dynamic sliding-window mechanism；
                # remainder1； slide； remainder2; initial_frame_num
                initial_frame_num_adjust = initial_frame_num - self.temporal_memory_len
                remainder1 = timesteps_iter_index % slide_size
                left_frame_len = frame_len - remainder1
                if left_frame_len < initial_frame_num_adjust:
                    remainder1 = 0
                    left_frame_len = frame_len - remainder1
                    
                chunk_num = int((left_frame_len-initial_frame_num_adjust)/slide_size) + 2 + 1
                remainder = (left_frame_len-initial_frame_num_adjust)%slide_size

                start_index = 0
                for chunk_index in range(chunk_num):
                    if chunk_index == 0:
                        start_index = 0
                        cal_len = remainder1 
                        if cal_len == 0:
                            continue
                    elif chunk_index == chunk_num-2:
                        start_index = frame_len - initial_frame_num_adjust - remainder
                        cal_len = remainder 
                        if cal_len == 0:
                            continue
                    elif chunk_index == chunk_num-1:
                        cal_len = initial_frame_num_adjust
                        start_index = frame_len - initial_frame_num_adjust
                    elif chunk_index > 0 and chunk_index < chunk_num-1:
                        start_index = remainder1 + (chunk_index-1)*slide_size
                        cal_len = slide_size
                    else:
                        print("wrong in initial stage2!!!!!!!!")
                    end_index = start_index + initial_frame_num_adjust
                    latent_model_input = latent_list[start_index:end_index].copy() 
                    t_item = timesteps_list[start_index:end_index].copy()

                    #### involve memory
                    if start_index < self.temporal_memory_len:
                        latent_model_input = latent_list[start_index:end_index+self.temporal_memory_len].copy()
                        t_item = timesteps_list[start_index:end_index+self.temporal_memory_len].copy()
                        index_bais = 0
                    else:
                        index_bais = self.temporal_memory_len
                        if start_index >= self.temporal_memory_len*saw_width:
                            memory_latents_list = []
                            memory_steps_list = []
                            for memory_index in reversed(range(self.temporal_memory_len)):
                                memory_latents_list.append(latent_list[start_index-(3 + (memory_index)*saw_width)])
                                memory_steps_list.append(timesteps_list[start_index-(3 + (memory_index)*saw_width)])
                                if start_index-(3 + (memory_index)*saw_width) < 0 or (3 + (memory_index)*saw_width) <0:
                                    print(f"!!!!!!!!!wrong memory_index,{start_index},{memory_index},{saw_width}")
                            latent_model_input = memory_latents_list+ latent_model_input
                            t_item = memory_steps_list + t_item

                        else:
                            memory_sample_slide = int(start_index/self.temporal_memory_len)
                            memory_latents_list = []
                            memory_steps_list = []
                            for memory_index in reversed(range(self.temporal_memory_len)):
                                memory_latents_list.append(latent_list[start_index- (memory_index+1)*memory_sample_slide])
                                memory_steps_list.append(timesteps_list[start_index-(memory_index+1)*memory_sample_slide])
                                if start_index- (memory_index+1)*memory_sample_slide < 0 or (memory_index+1)*memory_sample_slide < 0:
                                    print(f"!!!!!!!!!wrong memory_index,{start_index},{memory_index},{memory_sample_slide}")
                            latent_model_input = memory_latents_list+ latent_model_input
                            t_item = memory_steps_list + t_item

                    latent_model_input = [torch.cat(latent_model_input, dim=1)] 
                    t_item = torch.stack(t_item)
                    timestep = t_item                 

                    self.model.to(self.device)
                    noise_pred_cond = self.model(
                        latent_model_input, t=timestep, **arg_c)[0]
                    noise_pred_uncond = self.model(
                        latent_model_input, t=timestep, **arg_null)[0]
                    noise_pred = noise_pred_uncond + guide_scale * (
                        noise_pred_cond - noise_pred_uncond)     

                    ###  sample_by_frame_wise   
                    for frame_index in range(cal_len):
                        load_index = start_index + frame_index
                        given_dict = {}
                        given_dict["model_outputs"]= quene_memory_dict["model_outputs"][load_index].copy()
                        given_dict["timestep_list"]= quene_memory_dict["timestep_list"][load_index].copy()
                        given_dict["last_sample"]= quene_memory_dict["last_sample"][load_index]
                        given_dict["this_order"]= quene_memory_dict["this_order"][load_index]
                        

                        temp_x0_output_dict = sample_scheduler.step_by_different_e(
                            noise_pred[:,frame_index + index_bais ].unsqueeze(0).unsqueeze(2),       
                            timestep[frame_index + index_bais ],
                            latent_model_input[0][:,frame_index + index_bais ].unsqueeze(0).unsqueeze(2),    
                            given_dict=given_dict,
                            return_dict=False,
                            generator=seed_g)
                        
                        ### save & update
                        for key_item in  temp_x0_output_dict:
                            if isinstance(temp_x0_output_dict[key_item], list):
                                quene_memory_dict[key_item][load_index] = temp_x0_output_dict[key_item].copy()
                            else:
                                quene_memory_dict[key_item][load_index] = temp_x0_output_dict[key_item]
        
                ### Check whether the first frame has been fully denoised, and then save it.
                if steps_list[0] >= fifo_end_noise_index -1 and begin_pop_flag:
                    for chunk_index_inner in range(saw_width):
                        first_frame_index = 0
                        saved_latents.append(quene_memory_dict['prev_sample'][first_frame_index].squeeze(0) )

                        for key in quene_memory_dict:
                            if key not in pop_quene_memory_dict:
                                pop_quene_memory_dict[key] = [quene_memory_dict[key][0]]
                            else:
                                pop_quene_memory_dict[key].append(quene_memory_dict[key][0])
                            quene_memory_dict[key].pop(0)

                    
                    latent_list = []
                    for frame_item in quene_memory_dict['prev_sample']:
                        latent_list.append(frame_item.squeeze(0))
                    
                    steps_list = steps_list[:len(latent_list)]
                    timesteps_list = timesteps_list[:len(latent_list)]

                else:
                    for frame_index in range(len(latent_list)):
                        steps_list[frame_index] = steps_list[frame_index] + 1
                        timesteps_list[frame_index] = timesteps[  steps_list[frame_index] ]

                    latent_list = []
                    for frame_item in quene_memory_dict['prev_sample']:
                        if frame_item is not None:
                            latent_list.append(frame_item.squeeze(0))
                        else:
                            noise_item = torch.randn_like(latent_list[-1])
                            latents_item = latents_initial[0][:,-1].unsqueeze(1)
                            latents_item_with_noise0 = sample_scheduler.add_noise(latents_item, noise_item,timesteps[steps_list[-1]].unsqueeze(0))
                            latent_list.append( latents_item_with_noise0 ) 

                    if steps_list[0] >= fifo_end_noise_index-1:
                        begin_pop_flag = True
                        save_info = {}
                        save_info["latent_list"] = latent_list
                        save_info["steps_list"] = steps_list
                        save_info["timesteps_list"] = timesteps_list
                        save_info["quene_memory_dict"] = quene_memory_dict
                        save_info["saved_latents"] = saved_latents
                        save_info["begin_pop_flag"] = begin_pop_flag


                #### Determine whether to append a new noise frame.
                if begin_pop_flag and steps_list[-1] <=1:
                    for _ in range(saw_width):
                        steps_list.append(0)
                        timesteps_list.append(timesteps[0])
                        noise_item = torch.randn_like(latent_list[-1])
                        latents_item = latents_initial[0][:,-1].unsqueeze(1)
                        latents_item_with_noise0 = sample_scheduler.add_noise(latents_item, noise_item,timesteps[steps_list[-1]].unsqueeze(0))
                        latent_list.append( latents_item_with_noise0 ) 

                        quene_memory_dict["prev_sample"].append(None)
                        quene_memory_dict["model_outputs"].append([None,None])
                        quene_memory_dict["timestep_list"].append([None,None])
                        quene_memory_dict["last_sample"].append(None)
                        quene_memory_dict["this_order"].append(None)

            ## save res
            if long_generate_setting.save_inter_results:
                save_file = os.path.join(save_dir,"saved_video_fifo_p3.mp4")  
                self.decode_and_save_video(saved_latents, save_file)

            ############################################ Stage 2: Denoising at a Unified Noise Level. ############################################
            print("Stage 2: Denoising at a Unified Noise Level.")    
            latent_list = saved_latents
            timesteps_list, steps_list = [],[]
            noise_index =   long_generate_setting.sampling_steps - fifo_end_noise_index
            frame_len = len(latent_list)
            for frame_index in range(frame_len):
                timesteps_list.append(timesteps[ long_generate_setting.sampling_steps - noise_index-1]  )
                steps_list.append( long_generate_setting.sampling_steps - noise_index -1)

            quene_memory_dict = self.initial_quene_memory_dict(frame_len)

            for timesteps_iter_index in tqdm(range(long_generate_setting.sampling_steps - fifo_end_noise_index)): 
                frame_len = len(latent_list)
                #### To avoid jitter caused by a fixed sliding-window trajectory, we design a dynamic sliding-window mechanism；
                # remainder1； slide； remainder2; initial_frame_num
                initial_frame_num_adjust = initial_frame_num
                remainder1 = timesteps_iter_index % slide_size
                left_frame_len = frame_len - remainder1
                if left_frame_len < initial_frame_num_adjust:
                    remainder1 = 0
                    left_frame_len = frame_len - remainder1

                chunk_num = int((left_frame_len-initial_frame_num_adjust)/slide_size) + 2 + 1
                remainder = (left_frame_len-initial_frame_num_adjust)%slide_size

                for chunk_index in range(chunk_num):
                    if chunk_index == 0:
                        start_index = 0
                        cal_len = remainder1 
                        if cal_len == 0:
                            continue
                    elif chunk_index == chunk_num-2:
                        start_index = frame_len - initial_frame_num_adjust - remainder
                        cal_len = remainder 
                        if cal_len == 0:
                            continue
                    elif chunk_index == chunk_num-1:
                        cal_len = initial_frame_num_adjust
                        start_index = frame_len - initial_frame_num_adjust
                    elif chunk_index > 0 and chunk_index < chunk_num-1:
                        start_index = remainder1 + (chunk_index-1)*slide_size
                        cal_len = slide_size
                    else:
                        print("wrong in initial stage2!!!!!!!!")
                    end_index = start_index + initial_frame_num_adjust
                    
                    latent_model_input = latent_list[start_index:end_index].copy() 
                    t_item = timesteps_list[start_index:end_index].copy()

                    latent_model_input = [torch.cat(latent_model_input, dim=1)]
                    t_item = torch.stack(t_item)
                    timestep = t_item                 
                
                    self.model.to(self.device)
                    noise_pred_cond = self.model(
                        latent_model_input, t=timestep, **arg_c)[0]
                    noise_pred_uncond = self.model(
                        latent_model_input, t=timestep, **arg_null)[0]
                    noise_pred = noise_pred_uncond + guide_scale * (
                        noise_pred_cond - noise_pred_uncond)        

                    ###  sample_by_frame_wise         
                    for frame_index in range(cal_len):
                        load_index = start_index + frame_index
                        given_dict = {}
                        given_dict["model_outputs"]= quene_memory_dict["model_outputs"][load_index].copy()
                        given_dict["timestep_list"]= quene_memory_dict["timestep_list"][load_index].copy()
                        given_dict["last_sample"]= quene_memory_dict["last_sample"][load_index]
                        given_dict["this_order"]= quene_memory_dict["this_order"][load_index]
                        
                        temp_x0_output_dict = sample_scheduler.step_by_different_e(
                            noise_pred[:,frame_index ].unsqueeze(0).unsqueeze(2),                
                            timestep[frame_index ],
                            latent_model_input[0][:,frame_index ].unsqueeze(0).unsqueeze(2),       
                            given_dict=given_dict,
                            return_dict=False,
                            generator=seed_g)
                        
                        ### save & update
                        for key_item in  temp_x0_output_dict:
                            if isinstance(temp_x0_output_dict[key_item], list):
                                quene_memory_dict[key_item][load_index] = temp_x0_output_dict[key_item].copy()
                            else:
                                quene_memory_dict[key_item][load_index] = temp_x0_output_dict[key_item]
        
                ####### update steps_list， timesteps_list
                if steps_list[0] < long_generate_setting.sampling_steps - 1:
                    for frame_index in range(len(latent_list)):
                        steps_list[frame_index] = steps_list[frame_index] + 1
                        timesteps_list[frame_index] = timesteps[  steps_list[frame_index] ]
                latent_list = []
                for frame_item in quene_memory_dict['prev_sample']:
                    latent_list.append(frame_item.squeeze(0))

            ### add & re-denoise
            initial_latent_list =  latent_list.copy()

            for re_de_index in range(1):
                latent_list = []
                timesteps_list, steps_list = [],[]
                noise_index =   10
                frame_len = len(initial_latent_list)
                for frame_index in range(frame_len):
                    timesteps_list.append(timesteps[ sampling_steps - noise_index]  )
                    steps_list.append( sampling_steps - noise_index)
                    noise_item = torch.randn_like(initial_latent_list[-1])
                    latents_item_with_noise0 = sample_scheduler.add_noise(initial_latent_list[frame_index], noise_item,timesteps[sampling_steps -noise_index].unsqueeze(0))
                    latent_list.append(latents_item_with_noise0)

                quene_memory_dict = self.initial_quene_memory_dict(frame_len)

                for timesteps_iter_index in tqdm(range(noise_index)):
                    frame_len = len(latent_list)
                    #### To avoid jitter caused by a fixed sliding-window trajectory, we design a dynamic sliding-window mechanism；
                    # remainder1； slide； remainder2; initial_frame_num
                    initial_frame_num_adjust = initial_frame_num 
                    remainder1 = timesteps_iter_index % slide_size
                    left_frame_len = frame_len - remainder1
                    if left_frame_len < initial_frame_num_adjust:
                        remainder1 = 0
                        left_frame_len = frame_len - remainder1
                        
                    chunk_num = int((left_frame_len-initial_frame_num_adjust)/slide_size) + 2 + 1
                    remainder = (left_frame_len-initial_frame_num_adjust)%slide_size

                    for chunk_index in range(chunk_num):
                        if chunk_index == 0:
                            start_index = 0
                            cal_len = remainder1 
                            if cal_len == 0:
                                continue
                        elif chunk_index == chunk_num-2:
                            start_index = frame_len - initial_frame_num_adjust - remainder
                            cal_len = remainder 
                            if cal_len == 0:
                                continue
                        elif chunk_index == chunk_num-1:
                            cal_len = initial_frame_num_adjust
                            start_index = frame_len - initial_frame_num_adjust
                        elif chunk_index > 0 and chunk_index < chunk_num-1:
                            start_index = remainder1 + (chunk_index-1)*slide_size
                            cal_len = slide_size
                        else:
                            print("wrong in initial stage2!!!!!!!!")
                        end_index = start_index + initial_frame_num_adjust
                        
                        latent_model_input = latent_list[start_index:end_index].copy() 
                        t_item = timesteps_list[start_index:end_index].copy()

                        latent_model_input = [torch.cat(latent_model_input, dim=1)]
                        t_item = torch.stack(t_item)
                        timestep = t_item               
                        
                        self.model.to(self.device)
                        noise_pred_cond = self.model(
                            latent_model_input, t=timestep, **arg_c)[0]
                        noise_pred_uncond = self.model(
                            latent_model_input, t=timestep, **arg_null)[0]
                        noise_pred = noise_pred_uncond + guide_scale * (
                            noise_pred_cond - noise_pred_uncond)   

                        #### sample_by_frame_wise         
                        for frame_index in range(cal_len):
                            load_index = start_index + frame_index
                            given_dict = {}
                            given_dict["model_outputs"]= quene_memory_dict["model_outputs"][load_index].copy()
                            given_dict["timestep_list"]= quene_memory_dict["timestep_list"][load_index].copy()
                            given_dict["last_sample"]= quene_memory_dict["last_sample"][load_index]
                            given_dict["this_order"]= quene_memory_dict["this_order"][load_index]
                            
                            temp_x0_output_dict = sample_scheduler.step_by_different_e(
                                noise_pred[:,frame_index ].unsqueeze(0).unsqueeze(2),     
                                timestep[frame_index ],
                                latent_model_input[0][:,frame_index ].unsqueeze(0).unsqueeze(2),    
                                given_dict=given_dict,
                                return_dict=False,
                                generator=seed_g)
                            
                            ### save & update
                            for key_item in  temp_x0_output_dict: 
                                if isinstance(temp_x0_output_dict[key_item], list):
                                    quene_memory_dict[key_item][load_index] = temp_x0_output_dict[key_item].copy()
                                else:
                                    quene_memory_dict[key_item][load_index] = temp_x0_output_dict[key_item]
            
                    ####### update steps_list， timesteps_list
                    if steps_list[0] < long_generate_setting.sampling_steps - 1:
                        for frame_index in range(len(latent_list)):
                            steps_list[frame_index] = steps_list[frame_index] + 1
                            timesteps_list[frame_index] = timesteps[  steps_list[frame_index] ]
                    latent_list = []
                    for frame_item in quene_memory_dict['prev_sample']:
                        latent_list.append(frame_item.squeeze(0))

                ## save res
                # if long_generate_setting.save_inter_results:
                final_latent_list =latent_list[:-21]
                save_file = os.path.join(save_dir,"saved_video_final_long.mp4")  
                self.decode_and_save_video(final_latent_list, save_file)
