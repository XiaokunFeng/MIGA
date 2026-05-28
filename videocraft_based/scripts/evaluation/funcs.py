import os, sys, glob, math
import numpy as np
from collections import OrderedDict
from decord import VideoReader, cpu
import cv2
import torch
import torchvision
from PIL import Image
import PIL
import imageio
from tqdm import trange
sys.path.insert(1, os.path.join(sys.path[0], '..', '..'))
from lvdm.models.samplers.ddim import DDIMSampler
from tqdm import tqdm
from torch.nn import functional


def export_to_video(video_frames, output_video_path, fps):
    if isinstance(video_frames[0], np.ndarray):
        if video_frames[0].dtype != np.uint8:
            video_frames = [(frame * 255).astype(np.uint8) for frame in video_frames]

    elif isinstance(video_frames[0], PIL.Image.Image):
        video_frames = [np.array(frame) for frame in video_frames]

    with imageio.get_writer(output_video_path, fps=fps) as writer:
        for frame in video_frames:
            writer.append_data(frame)
def get_filelist(data_dir, ext='*'):
    file_list = glob.glob(os.path.join(data_dir, '*.%s'%ext))
    file_list.sort()
    return file_list

def get_dirlist(path):
    list = []
    if (os.path.exists(path)):
        files = os.listdir(path)
        for file in files:
            m = os.path.join(path,file)
            if (os.path.isdir(m)):
                list.append(m)
    list.sort()
    return list


def load_model_checkpoint(model, ckpt):
    def load_checkpoint(model, ckpt, full_strict):
        state_dict = torch.load(ckpt, map_location="cpu")
        try:
            ## deepspeed
            new_pl_sd = OrderedDict()
            for key in state_dict['module'].keys():
                new_pl_sd[key[16:]]=state_dict['module'][key]
            model.load_state_dict(new_pl_sd, strict=full_strict)
        except:
            if "state_dict" in list(state_dict.keys()):
                state_dict = state_dict["state_dict"]
            model.load_state_dict(state_dict, strict=full_strict)
        return model
    load_checkpoint(model, ckpt, full_strict=True)
    print('>>> model checkpoint loaded.')
    return model


def load_prompts(prompt_file):
    f = open(prompt_file, 'r')
    prompt_list = []
    for idx, line in enumerate(f.readlines()):
        l = line.strip()
        if len(l) != 0:
            prompt_list.append(l)
        f.close()
    return prompt_list


def load_video_batch(filepath_list, frame_stride, video_size=(256,256), video_frames=16):
    '''
    Notice about some special cases:
    1. video_frames=-1 means to take all the frames (with fs=1)
    2. when the total video frames is less than required, padding strategy will be used (repreated last frame)
    '''
    fps_list = []
    batch_tensor = []
    assert frame_stride > 0, "valid frame stride should be a positive interge!"
    for filepath in filepath_list:
        padding_num = 0
        vidreader = VideoReader(filepath, ctx=cpu(0), width=video_size[1], height=video_size[0])
        fps = vidreader.get_avg_fps()
        total_frames = len(vidreader)
        max_valid_frames = (total_frames-1) // frame_stride + 1
        if video_frames < 0:
            ## all frames are collected: fs=1 is a must
            required_frames = total_frames
            frame_stride = 1
        else:
            required_frames = video_frames
        query_frames = min(required_frames, max_valid_frames)
        frame_indices = [frame_stride*i for i in range(query_frames)]

        ## [t,h,w,c] -> [c,t,h,w]
        frames = vidreader.get_batch(frame_indices)
        frame_tensor = torch.tensor(frames.asnumpy()).permute(3, 0, 1, 2).float()
        frame_tensor = (frame_tensor / 255. - 0.5) * 2
        if max_valid_frames < required_frames:
            padding_num = required_frames - max_valid_frames
            frame_tensor = torch.cat([frame_tensor, *([frame_tensor[:,-1:,:,:]]*padding_num)], dim=1)
            print(f'{os.path.split(filepath)[1]} is not long enough: {padding_num} frames padded.')
        batch_tensor.append(frame_tensor)
        sample_fps = int(fps/frame_stride)
        fps_list.append(sample_fps)
    
    return torch.stack(batch_tensor, dim=0)

def load_image_batch(filepath_list, image_size=(256,256)):
    batch_tensor = []
    for filepath in filepath_list:
        _, filename = os.path.split(filepath)
        _, ext = os.path.splitext(filename)
        if ext == '.mp4':
            vidreader = VideoReader(filepath, ctx=cpu(0), width=image_size[1], height=image_size[0])
            frame = vidreader.get_batch([0])
            img_tensor = torch.tensor(frame.asnumpy()).squeeze(0).permute(2, 0, 1).float()
        elif ext == '.png' or ext == '.jpg':
            img = Image.open(filepath).convert("RGB")
            rgb_img = np.array(img, np.float32)
            #bgr_img = cv2.imread(filepath, cv2.IMREAD_COLOR)
            #bgr_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
            rgb_img = cv2.resize(rgb_img, (image_size[1],image_size[0]), interpolation=cv2.INTER_LINEAR)
            img_tensor = torch.from_numpy(rgb_img).permute(2, 0, 1).float()
        else:
            print(f'ERROR: <{ext}> image loading only support format: [mp4], [png], [jpg]')
            raise NotImplementedError
        img_tensor = (img_tensor / 255. - 0.5) * 2
        batch_tensor.append(img_tensor)
    return torch.stack(batch_tensor, dim=0)


def save_videos(batch_tensors, savedir, filenames, fps=10):
    # b,samples,c,t,h,w
    n_samples = batch_tensors.shape[1]
    for idx, vid_tensor in enumerate(batch_tensors):
        video = vid_tensor.detach().cpu()
        video = torch.clamp(video.float(), -1., 1.)
        video = video.permute(2, 0, 1, 3, 4) # t,n,c,h,w
        frame_grids = [torchvision.utils.make_grid(framesheet, nrow=int(n_samples)) for framesheet in video] #[3, 1*h, n*w]
        grid = torch.stack(frame_grids, dim=0) # stack in temporal dim [t, 3, n*h, w]
        grid = (grid + 1.0) / 2.0
        grid = (grid * 255).to(torch.uint8).permute(0, 2, 3, 1) # [t, n*h, w, 3]
        savepath = os.path.join(savedir, f"{filenames[idx]}.mp4")
        torchvision.io.write_video(savepath, grid, fps=fps, video_codec='h264', options={'crf': '10'})

def save_gif(batch_tensors, savedir, name):
    vid_tensor = torch.squeeze(batch_tensors) # c,f,h,w

    video = vid_tensor.detach().cpu()
    video = torch.clamp(video.float(), -1., 1.)
    video = video.permute(1, 0, 2, 3) # f,c,h,w

    video = (video + 1.0) / 2.0
    video = (video * 255).to(torch.uint8).permute(0, 2, 3, 1) # f,h,w,c

    frames = video.chunk(video.shape[0], dim=0)
    frames = [frame.squeeze(0) for frame in frames]
    savepath = os.path.join(savedir, f"{name}.gif")

    imageio.mimsave(savepath, frames, duration=100)

def tensor2image(batch_tensors):
    img_tensor = torch.squeeze(batch_tensors) # c,h,w

    image = img_tensor.detach().cpu()
    image = torch.clamp(image.float(), -1., 1.)

    image = (image + 1.0) / 2.0
    image = (image * 255).to(torch.uint8).permute(1, 2, 0) # h,w,c
    image = image.numpy()
    image = Image.fromarray(image)
    
    return image


def adjust_judge(saved_latents,new_gen_latents,sim_result,threshold=0.001): # 0.075. 0.008 0.003
    adjust_flag = False
    feats = saved_latents
    feats = torch.cat(feats,dim=1).squeeze(2)
    f,d,h,w = feats.shape
    feats = feats.view(f,d,-1) # f,16,6240 -> f,6240,16
    frame_vecs = feats.mean(dim=1)  
    frame_vecs_1 = functional.normalize(frame_vecs, p=2, dim=1)

    feats = new_gen_latents
    feats = torch.cat(feats,dim=1).squeeze(2)
    f,d,h,w = feats.shape
    feats = feats.view(f,d,-1) # f,16,6240 -> f,6240,16
    frame_vecs = feats.mean(dim=1)  
    frame_vecs_2 = functional.normalize(frame_vecs, p=2, dim=1)

    sim_matrix = frame_vecs_1 @ frame_vecs_2.T
    sim_matrix_weight = sim_matrix.mean(0).mean(0)

    if len(sim_result) > 0:
        latest_sim = sim_result[-1]
        sim_result.append(sim_matrix_weight) 
        if latest_sim - sim_matrix_weight> threshold:
            ##### 开始调整
            adjust_flag = True
            print(f"begin adjusting, sim_result: {sim_result}")
            
    else:
        sim_result.append(sim_matrix_weight) 
    return adjust_flag, sim_result   
    


##### for generate_v4_4_involve_memory_and_sample
def prepare_latents_by_step_initialize_with_temporal_memory_and_resample(args, latents_dir, sampler,cond,noise_shape,cfg_scale,uc,long_generate_setting):
    video = torch.load(latents_dir+f"/{long_generate_setting.sampling_steps}.pt") 
    alphas = sampler.ddim_alphas      # 0.999--> 0.005; sampler.ddim_timesteps： 0——> 999
    timesteps = sampler.ddim_timesteps         # [0, 16, 32,...., 999]
    indices = np.arange(long_generate_setting.sampling_steps)

    ############  process_0: Initialize the last initial_temporal_len latents using the existing initial_temporal_len latents. 
    saw_width = long_generate_setting.saw_width
    saw_height = 1
    long_latent_len = long_generate_setting.sampling_steps*saw_width
    initial_temporal_len = video.shape[2]
    initial_frame_num = video.shape[2]
    noise_index = int((long_latent_len - initial_temporal_len)/saw_width) - 1
    frame_index = 0

    latent_list,timesteps_list, steps_list = [],[],[]
    for chunk_index in range(int(initial_temporal_len/saw_width)):
        for chunk_index_inner in range(saw_width):
            latents_item  = video[:,:,chunk_index*saw_width+chunk_index_inner].unsqueeze(2)

            noise_item = torch.randn_like(latents_item) 
            timesteps_list.append(timesteps[ noise_index]  )
            steps_list.append( noise_index )

            latents_item_with_noise0 = sampler.add_noise(latents_item, noise_item,noise_index)
            latent_list.append(latents_item_with_noise0)
        noise_index += saw_height

    #############  process_1: Progressively complete the initialization of the remaining latent frames.
    noise_index = long_generate_setting.sampling_steps - 1  

     ###### involve memory 
    temporal_memory_len = long_generate_setting.temporal_memory_len 

    initial_frame_num_adjust = initial_frame_num - temporal_memory_len
    initial_iter_nums =  long_generate_setting.long_iter_nums 
    if initial_iter_nums > long_latent_len/saw_width - initial_frame_num/saw_width:
        initial_iter_nums = int(long_latent_len/saw_width - initial_frame_num/saw_width)
    
    sim_result = []

    fifo_end_noise_index = 10
    for timesteps_iter_index in tqdm(range( initial_iter_nums )): 
        for chunk_index_inner in range(saw_width):
            latents_item  = video[:,:,-1].unsqueeze(2)
            noise_item = torch.randn_like(latents_item) 
            timesteps_list.append(timesteps[ noise_index]  )
            steps_list.append(   noise_index )

            latents_item_with_noise0 = sampler.add_noise(latents_item, noise_item,noise_index)
            latent_list.append(latents_item_with_noise0)

        if len(latent_list) == long_latent_len:
            break

        if steps_list[0] <= fifo_end_noise_index+1:
            break
        
        ####  determine the chunk infor
        frame_len = len(latent_list)
        slide_size = 8

        chunk_num = int((frame_len-initial_frame_num_adjust)/slide_size) + 2
        remainder = (frame_len-initial_frame_num_adjust)%slide_size
        
        for saw_index in range(saw_height):
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
                    print("wrong in initial stage2!!!!!!!!")
                end_index = start_index + initial_frame_num_adjust

                latent_model_input = latent_list[start_index:end_index].copy() 
                t = timesteps_list[start_index:end_index]
                idx = steps_list[start_index:end_index] 
                
                #### involve memory
                if start_index < temporal_memory_len:
                    ## Retrieve latents from later
                    latent_model_input = latent_list[start_index:end_index+temporal_memory_len].copy()
                    t = timesteps_list[start_index:end_index+temporal_memory_len].copy()
                    idx  = steps_list[start_index:end_index+temporal_memory_len].copy()
                    index_bais = 0
                else:
                    index_bais = temporal_memory_len
                    if start_index >= temporal_memory_len*saw_width:
                        memory_latents_list = []
                        memory_steps_list = []
                        memory_idxs_list = []
                        chunk_iter_index = int(saw_width/2)
                        for memory_index in reversed(range(temporal_memory_len)):
                            memory_latents_list.append(latent_list[start_index-(chunk_iter_index + (memory_index)*saw_width)])
                            memory_steps_list.append(timesteps_list[start_index-(chunk_iter_index + (memory_index)*saw_width)])
                            memory_idxs_list.append(steps_list[start_index-(chunk_iter_index + (memory_index)*saw_width)])
                            if start_index-(chunk_iter_index + (memory_index)*saw_width) < 0 or (chunk_iter_index + (memory_index)*saw_width) <0:
                                print(f"!!!!!!!!!wrong memory_index,{start_index},{memory_index},{saw_width}")
                        latent_model_input = memory_latents_list+ latent_model_input
                        t = memory_steps_list + t
                        idx = memory_idxs_list + idx

                    else:
                        memory_sample_slide = int(start_index/temporal_memory_len)
                        memory_latents_list = []
                        memory_steps_list = []
                        memory_idxs_list = []
                        for memory_index in reversed(range(temporal_memory_len)):
                            memory_latents_list.append(latent_list[start_index- (memory_index+1)*memory_sample_slide])
                            memory_steps_list.append(timesteps_list[start_index-(memory_index+1)*memory_sample_slide])
                            memory_idxs_list.append(steps_list[start_index-(memory_index+1)*memory_sample_slide])
                            if start_index- (memory_index+1)*memory_sample_slide < 0 or (memory_index+1)*memory_sample_slide < 0:
                                print(f"!!!!!!!!!wrong memory_index,{start_index},{memory_index},{memory_sample_slide}")
                        latent_model_input = memory_latents_list+ latent_model_input
                        t = memory_steps_list + t
                        idx = memory_idxs_list + idx

                input_latents = torch.cat(latent_model_input, dim=2)
                output_latents, _ = sampler.fifo_onestep(
                                cond=cond,
                                shape=noise_shape,              # [1, 4, 16, 40, 64]
                                latents=input_latents,          # ([1, 4, 16, 40, 64])
                                timesteps=t,
                                indices=idx,
                                unconditional_guidance_scale=cfg_scale,
                                unconditional_conditioning=uc
                                )

                for frame_index in range(cal_len):
                    ### load saved infor
                    load_index = start_index + frame_index

                    ### update infor
                    latent_list[load_index] = output_latents[:,:,[frame_index + index_bais]]
                    steps_list[load_index] = steps_list[load_index] - 1
                    timesteps_list[load_index] = timesteps[  steps_list[load_index] ]
                del output_latents

        
        ############# resample process (reflection and correction processing)
        judged_step = 50 # 50
        guide_chunk_num = 3   #  if saw_width >= 4 else 15
        if steps_list[0] < judged_step - guide_chunk_num and long_generate_setting.involve_resample:
            judged_index = int(judged_step - steps_list[0])
            judged_latents = latent_list[judged_index*saw_width:(judged_index+1)*saw_width]
            judged_steps = steps_list[judged_index*saw_width:(judged_index+1)*saw_width]
            early_latents = latent_list[(judged_index-guide_chunk_num)*saw_width:judged_index*saw_width]

            adjust_flag,sim_result =adjust_judge(judged_latents,early_latents,sim_result)
            print(f"%%%%%%%%%%%%% sim_result_latents:{sim_result}  at steps_list[0]: {steps_list[0]}")

            if adjust_flag == True:
                guide_latents = early_latents.copy()
                guide_steps = steps_list[(judged_index-guide_chunk_num)*saw_width:(judged_index)*saw_width]
                guide_timesteps = timesteps_list[(judged_index-guide_chunk_num)*saw_width:(judged_index)*saw_width]
                
                ### Progressively guide generation
                adjust_sample_res = []
                for adjust_sample_index in range(5):
                    adjust_latents_list = guide_latents.copy()
                    adjust_steps_list = guide_steps.copy()
                    adjust_timesteps_list = guide_timesteps.copy()
                    
                    for adjust_timesteps_iter_index in tqdm(range( args.num_inference_steps - judged_step-1)):
                        for chunk_index_inner in range(saw_width):
                            noise_index = args.num_inference_steps - 1
                            latents_item  = video[:,:,-1].unsqueeze(2)
                            noise_item = torch.randn_like(latents_item)
                            adjust_timesteps_list.append(timesteps[ noise_index]  )
                            adjust_steps_list.append(   noise_index )

                            latents_item_with_noise0 = sampler.add_noise(latents_item, noise_item,noise_index)
                            adjust_latents_list.append(latents_item_with_noise0)

                        ####  determine the chunk infor
                        adjust_frame_len = len(adjust_latents_list)
                        slide_size = 8

                        chunk_num = int((adjust_frame_len-initial_frame_num)/slide_size) + 2 if adjust_frame_len-initial_frame_num > 0 else 1
                        remainder = (adjust_frame_len-initial_frame_num)%slide_size if adjust_frame_len-initial_frame_num > 0 else adjust_frame_len

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

                            ### 
                            latent_model_input = adjust_latents_list[start_index:end_index].copy()      # latent_list:  41 x ( ([16, 1, 60, 104])  )
                            t = adjust_timesteps_list[start_index:end_index]
                            idx = adjust_steps_list[start_index:end_index] 
                            

                            input_latents = torch.cat(latent_model_input, dim=2)
                            output_latents, _ = sampler.fifo_onestep(
                                            cond=cond,
                                            shape=noise_shape,              # [1, 4, 16, 40, 64]
                                            latents=input_latents,          # ([1, 4, 16, 40, 64])
                                            timesteps=t,
                                            indices=idx,
                                            unconditional_guidance_scale=cfg_scale,
                                            unconditional_conditioning=uc,
                                            )
                            ###############  sample_by_frame_wise 
                            for frame_index in range(cal_len):
                                ### load saved infor
                                load_index = start_index + frame_index
                                if load_index < guide_chunk_num*saw_width:
                                    continue
                                ### update infor
                                adjust_latents_list[load_index] = output_latents[:,:,[frame_index]]
                                adjust_steps_list[load_index] = adjust_steps_list[load_index] - 1
                                adjust_timesteps_list[load_index] = timesteps[  adjust_steps_list[load_index] ]

                    
                    adjust_sample_res.append(adjust_latents_list)
                
                ##### update
                max_sim_res,max_sim_index = -10,0
                for adjust_sample_index in range(len(adjust_sample_res)):
                    adjust_latents_list = adjust_sample_res[adjust_sample_index]
                    new_gen_latents = adjust_latents_list[guide_chunk_num*saw_width: (guide_chunk_num + 1)*saw_width]
                
                    _, sim_result_1 = adjust_judge(guide_latents,new_gen_latents,[])
                    if sim_result_1[-1] > max_sim_res:
                        max_sim_res = sim_result_1[-1]
                        max_sim_index = adjust_sample_index
                print(f" >>>>>>>>>>>>>>>>>>>>>>>>>>>> saved sim_res:{sim_result}, sim_result_1: {sim_result_1},max_sim_res: {max_sim_res},max_sim_index:{max_sim_index}")
                if max_sim_res  > sim_result[-1]:
                    print(f"************************** adjust")
                    ###### save the final res
                    adjust_latents_list = adjust_sample_res[max_sim_index]
                    ### latent_list
                    latent_list_0 = latent_list[:(judged_index)*saw_width].copy()
                    latent_list_1 = adjust_latents_list[guide_chunk_num*saw_width:].copy()
                    latent_list = latent_list_0 + latent_list_1

                    sim_result[-1] = max_sim_res

                    #### steps_list just for check
                    steps_list_0 = steps_list[:(judged_index)*saw_width].copy()
                    steps_list_1 = adjust_steps_list[guide_chunk_num*saw_width:].copy()
                    steps_list = steps_list_0 + steps_list_1

    return latent_list,steps_list,timesteps_list


def shift_latents(latents):
    # shift latents
    latents[:,:,:-1] = latents[:,:,1:].clone()

    # add new noise to the last frame
    latents[:,:,-1] = torch.randn_like(latents[:,:,-1])

    return latents



def base_ddim_sampling(model, cond, noise_shape, ddim_steps=50, ddim_eta=1.0,\
                        cfg_scale=1.0, temporal_cfg_scale=None, latents_dir=None, **kwargs):
    ddim_sampler = DDIMSampler(model)
    uncond_type = model.uncond_type
    batch_size = noise_shape[0]
    ## construct unconditional guidance
    if cfg_scale != 1.0:
        if uncond_type == "empty_seq": # True
            prompts = batch_size * [""]
            #prompts = N * T * [""]  ## if is_imgbatch=True
            uc_emb = model.get_learned_conditioning(prompts)
        elif uncond_type == "zero_embed":
            c_emb = cond["c_crossattn"][0] if isinstance(cond, dict) else cond
            uc_emb = torch.zeros_like(c_emb)
                
        ## process image embedding token
        if hasattr(model, 'embedder'): # False
            uc_img = torch.zeros(noise_shape[0],3,224,224).to(model.device)
            ## img: b c h w >> b l c
            uc_img = model.get_image_embeds(uc_img)
            uc_emb = torch.cat([uc_emb, uc_img], dim=1)
        
        if isinstance(cond, dict): # True
            uc = {key:cond[key] for key in cond.keys()}
            uc.update({'c_crossattn': [uc_emb]})
        else: # False
            uc = uc_emb
    else:
        uc = None
    
    x_T = None

    if ddim_sampler is not None:
        kwargs.update({"clean_cond": True})
        samples, _ = ddim_sampler.sample(S=ddim_steps,                      # 64
                                        conditioning=cond,                  # cond['c_crossattn'][0].shape torch.Size([1, 77, 1024])
                                        batch_size=noise_shape[0],
                                        shape=noise_shape[1:],
                                        verbose=True,
                                        unconditional_guidance_scale=cfg_scale,
                                        unconditional_conditioning=uc,
                                        eta=ddim_eta,
                                        temporal_length=noise_shape[2],
                                        conditional_guidance_scale_temporal=temporal_cfg_scale,     # None
                                        x_T=x_T,                                                    # None
                                        latents_dir=latents_dir,
                                        **kwargs
                                        )
    ## reconstruct from latent to pixel space
    # samples: b,c,f,h,w        ——> ([1, 4, 16, 40, 64])
    batch_images = model.decode_first_stage_2DAE(samples) # b,c,f,H,W    ([1, 3, 16, 320, 512])

    return batch_images, ddim_sampler, samples


def generate_by_miga(args, model, conditioning, noise_shape, ddim_sampler,\
                        cfg_scale=1.0, save_dir=None,long_generate_setting = None, **kwargs):
    """
    MIGA
    """
    #### initial parameters
    batch_size = noise_shape[0]   
    kwargs.update({"clean_cond": True})
    # check condition bs
    if conditioning is not None:
        if isinstance(conditioning, dict):
            try:
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
            except:
                cbs = conditioning[list(conditioning.keys())[0]][0].shape[0]

            if cbs != batch_size:
                print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
        else:
            if conditioning.shape[0] != batch_size:
                print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")
    
    cond = conditioning

    ## construct unconditional guidance
    if cfg_scale != 1.0:
        prompts = batch_size * [""]
        #prompts = N * T * [""]  ## if is_imgbatch=True
        uc_emb = model.get_learned_conditioning(prompts)
        
        uc = {key:cond[key] for key in cond.keys()}
        uc.update({'c_crossattn': [uc_emb]})
        
    else:
        uc = None

    ############################################  initialize the quene   ############################################ 
    print(f"initialize the quene ...")
    latent_list,steps_list,timesteps_list = prepare_latents_by_step_initialize_with_temporal_memory_and_resample(args, save_dir, ddim_sampler, 
                                                    cond,noise_shape,cfg_scale,uc,long_generate_setting)      # ([1, 4, 72, 40, 64])

    ############################################ Stage 1: Zigzag Iterative Denoising. ############################################  
    print("Stage 1: Zigzag Iterative Denoising.")     
    fifo_video_frames = []
    saved_latents = []
    slide_size = 8
    saw_width = long_generate_setting.saw_width
    begin_pop_flag = False
    timesteps = ddim_sampler.ddim_timesteps
    initial_frame_num = args.video_length
    fifo_end_noise_index = 10 # steps of stage 2

    ###### involve memory 
    temporal_memory_len = long_generate_setting.temporal_memory_len
    initial_frame_num_adjust = initial_frame_num - temporal_memory_len

    timesteps_iter_index = -1
    while True:
        timesteps_iter_index += 1
        print(f"############################ timesteps_iter_index: {timesteps_iter_index} ############################ ")
        ### break condition
        if len(saved_latents) >= long_generate_setting.long_iter_nums*saw_width:    # 
            break

        frame_len = len(latent_list)
        #### To avoid jitter caused by a fixed sliding-window trajectory, we design a dynamic sliding-window mechanism；
        # remainder1； slide； remainder2; initial_frame_num
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

            latent_model_input = latent_list[start_index:end_index].copy()      # latent_list:  41 x ( ([16, 1, 60, 104])  )
            t = timesteps_list[start_index:end_index]
            idx = steps_list[start_index:end_index] 

            #### involve memory
            if start_index < temporal_memory_len:
                latent_model_input = latent_list[start_index:end_index+temporal_memory_len].copy()
                t = timesteps_list[start_index:end_index+temporal_memory_len].copy()
                idx  = steps_list[start_index:end_index+temporal_memory_len].copy()
                index_bais = 0
            else:
                index_bais = temporal_memory_len
                if start_index >= temporal_memory_len*saw_width:
                    memory_latents_list = []
                    memory_steps_list = []
                    memory_idxs_list = []
                    chunk_iter_index = int(saw_width/2)
                    for memory_index in reversed(range(temporal_memory_len)):
                        memory_latents_list.append(latent_list[start_index-(chunk_iter_index + (memory_index)*saw_width)])
                        memory_steps_list.append(timesteps_list[start_index-(chunk_iter_index + (memory_index)*saw_width)])
                        memory_idxs_list.append(steps_list[start_index-(chunk_iter_index + (memory_index)*saw_width)])
                        if start_index-(chunk_iter_index + (memory_index)*saw_width) < 0 or (chunk_iter_index + (memory_index)*saw_width) <0:
                            print(f"!!!!!!!!!wrong memory_index,{start_index},{memory_index},{saw_width}")
                    latent_model_input = memory_latents_list+ latent_model_input
                    t = memory_steps_list + t
                    idx = memory_idxs_list + idx

                else:
                    memory_sample_slide = int(start_index/temporal_memory_len)
                    memory_latents_list = []
                    memory_steps_list = []
                    memory_idxs_list = []
                    for memory_index in reversed(range(temporal_memory_len)):
                        memory_latents_list.append(latent_list[start_index- (memory_index+1)*memory_sample_slide])
                        memory_steps_list.append(timesteps_list[start_index-(memory_index+1)*memory_sample_slide])
                        memory_idxs_list.append(steps_list[start_index-(memory_index+1)*memory_sample_slide])
                        if start_index- (memory_index+1)*memory_sample_slide < 0 or (memory_index+1)*memory_sample_slide < 0:
                            print(f"!!!!!!!!!wrong memory_index,{start_index},{memory_index},{memory_sample_slide}")
                    latent_model_input = memory_latents_list+ latent_model_input
                    t = memory_steps_list + t
                    idx = memory_idxs_list + idx

            input_latents = torch.cat(latent_model_input, dim=2)
            output_latents, _ = ddim_sampler.fifo_onestep(
                            cond=cond,
                            shape=noise_shape,              # [1, 4, 16, 40, 64]
                            latents=input_latents,          # ([1, 4, 16, 40, 64])
                            timesteps=t,
                            indices=idx,
                            unconditional_guidance_scale=cfg_scale,
                            unconditional_conditioning=uc,
                            **kwargs
                            )
            ###  sample_by_frame_wise   
            for frame_index in range(cal_len):
                load_index = start_index + frame_index
                latent_list[load_index] = output_latents[:,:,[frame_index + index_bais]]
                steps_list[load_index] = steps_list[load_index] - 1
                timesteps_list[load_index] = timesteps[  steps_list[load_index] ]
            del output_latents

        ### Check whether the first frame has been fully denoised, and then save it.
        if steps_list[0] <= fifo_end_noise_index and begin_pop_flag:
            new_generated_frames = []
            for _ in range(saw_width):
                new_generated_frames.append(latent_list[0] )

                latent_list.pop(0)
            
            saved_latents.extend(new_generated_frames)

            steps_list = steps_list[saw_width:]
            timesteps_list = timesteps_list[saw_width:]

            for frame_item in new_generated_frames:
                frame_tensor = model.decode_first_stage_2DAE(frame_item)     # b,c,1,H,W
                image = tensor2image(frame_tensor)
                fifo_video_frames.append(image)
        else:
            if steps_list[0] <= fifo_end_noise_index:
                begin_pop_flag = True
      
                save_info = {}
                save_info["latent_list"] = latent_list
                save_info["steps_list"] = steps_list
                save_info["timesteps_list"] = timesteps_list
                save_info["saved_latents"] = saved_latents
                save_info["fifo_video_frames"] = fifo_video_frames
                save_info["begin_pop_flag"] = begin_pop_flag
                save_path = os.path.join(save_dir,"latent_fifo_p2.pt") 
                torch.save(save_info, save_path)

        ### Determine whether to append a new noise frame.
        if steps_list[-1]<= 62:
            noise_index = 63
            for _ in range(saw_width):
                timesteps_list.append(timesteps[ noise_index]  )
                steps_list.append(   noise_index )
                latent_list.append(torch.randn_like(latent_list[-1]))
       

    if long_generate_setting.save_inter_results:
        save_file = os.path.join(save_dir,f"saved_video_fifo_stage_long.mp4")
        export_to_video(fifo_video_frames, save_file, args.output_fps)

    
    ############################################ Stage 2: Denoising at a Unified Noise Level. ############################################
    print("Stage 2: Denoising at a Unified Noise Level.")   
    latent_list = saved_latents
    timesteps_list, steps_list = [],[]
    noise_index =   fifo_end_noise_index -1
    frame_len = len(latent_list)
    for frame_index in range(frame_len):
        timesteps_list.append(timesteps[ noise_index]  )
        steps_list.append( noise_index )

    for timesteps_iter_index in tqdm(range(noise_index)):   
        #### To avoid jitter caused by a fixed sliding-window trajectory, we design a dynamic sliding-window mechanism；
        # remainder1； slide； remainder2; initial_frame_num
        remainder1 = timesteps_iter_index % slide_size
        left_frame_len = frame_len - remainder1
        if left_frame_len < initial_frame_num:
            remainder1 = 0
            left_frame_len = frame_len - remainder1

        chunk_num = int((left_frame_len-initial_frame_num)/slide_size) + 2 + 1
        remainder = (left_frame_len-initial_frame_num)%slide_size

        start_index = 0
        for chunk_index in range(chunk_num):
            if chunk_index == 0:
                start_index = 0
                cal_len = remainder1 
                if cal_len == 0:
                    continue
            elif chunk_index == chunk_num-2:
                start_index = frame_len - initial_frame_num - remainder
                cal_len = remainder 
                if cal_len == 0:
                    continue
            elif chunk_index == chunk_num-1:
                cal_len = initial_frame_num
                start_index = frame_len - initial_frame_num
            elif chunk_index > 0 and chunk_index < chunk_num-1:
                start_index = remainder1 + (chunk_index-1)*slide_size
                cal_len = slide_size
            else:
                print("wrong in initial stage2!!!!!!!!")
            end_index = start_index + initial_frame_num

            latent_model_input = latent_list[start_index:end_index].copy()   
            t = timesteps_list[start_index:end_index]
            idx = steps_list[start_index:end_index] 

            input_latents = torch.cat(latent_model_input, dim=2)
            output_latents, _ = ddim_sampler.fifo_onestep(
                            cond=cond,
                            shape=noise_shape,              # [1, 4, 16, 40, 64]
                            latents=input_latents,          # ([1, 4, 16, 40, 64])
                            timesteps=t,
                            indices=idx,
                            unconditional_guidance_scale=cfg_scale,
                            unconditional_conditioning=uc,
                            **kwargs
                            )

            ###  sample_by_frame_wise 
            for frame_index in range(cal_len):
                load_index = start_index + frame_index
                latent_list[load_index] = output_latents[:,:,[frame_index]]
                steps_list[load_index] = steps_list[load_index] - 1
                timesteps_list[load_index] = timesteps[  steps_list[load_index] ]
            del output_latents


    ### add & re-denoise
    initial_latent_list = latent_list.copy()

    for redenoise_index in range(1):
        latent_list = []
        timesteps_list, steps_list = [],[]
        noise_index =   10
        frame_len = len(initial_latent_list)
        
        for frame_index in range(frame_len):
            timesteps_list.append(timesteps[ noise_index]  )
            steps_list.append( noise_index )
            latents_item  = initial_latent_list[frame_index]
            noise_item = torch.randn_like(latents_item)  
            latents_item_with_noise0 = ddim_sampler.add_noise(latents_item, noise_item,noise_index)
            latent_list.append(latents_item_with_noise0)

        for timesteps_iter_index in tqdm(range(noise_index)):
            #### To avoid jitter caused by a fixed sliding-window trajectory, we design a dynamic sliding-window mechanism；
            # remainder1； slide； remainder2; initial_frame_num
            remainder1 = timesteps_iter_index % slide_size
            left_frame_len = frame_len - remainder1
            if left_frame_len < initial_frame_num:
                remainder1 = 0
                left_frame_len = frame_len - remainder1

            chunk_num = int((left_frame_len-initial_frame_num)/slide_size) + 2 + 1
            remainder = (left_frame_len-initial_frame_num)%slide_size

            start_index = 0
            for chunk_index in range(chunk_num):
                if chunk_index == 0:
                    start_index = 0
                    cal_len = remainder1 
                    if cal_len == 0:
                        continue
                elif chunk_index == chunk_num-2:
                    start_index = frame_len - initial_frame_num - remainder
                    cal_len = remainder 
                    if cal_len == 0:
                        continue
                elif chunk_index == chunk_num-1:
                    cal_len = initial_frame_num
                    start_index = frame_len - initial_frame_num
                elif chunk_index > 0 and chunk_index < chunk_num-1:
                    start_index = remainder1 + (chunk_index-1)*slide_size
                    cal_len = slide_size
                else:
                    print("wrong in initial stage2!!!!!!!!")
                end_index = start_index + initial_frame_num

                latent_model_input = latent_list[start_index:end_index].copy()      # latent_list:  41 x ( ([16, 1, 60, 104])  )
                t = timesteps_list[start_index:end_index]
                idx = steps_list[start_index:end_index] 

                input_latents = torch.cat(latent_model_input, dim=2)
                output_latents, _ = ddim_sampler.fifo_onestep(
                                cond=cond,
                                shape=noise_shape,              # [1, 4, 16, 40, 64]
                                latents=input_latents,          # ([1, 4, 16, 40, 64])
                                timesteps=t,
                                indices=idx,
                                unconditional_guidance_scale=cfg_scale,
                                unconditional_conditioning=uc,
                                **kwargs
                                )

                for frame_index in range(cal_len):
                    load_index = start_index + frame_index

                    latent_list[load_index] = output_latents[:,:,[frame_index]]
                    steps_list[load_index] = steps_list[load_index] - 1
                    timesteps_list[load_index] = timesteps[  steps_list[load_index] ]
                del output_latents
            
        initial_latent_list = latent_list.copy()


    ## save the final video
    fifo_video_frames = []
    for frame_item in latent_list:
        frame_tensor = model.decode_first_stage_2DAE(frame_item)     # b,c,1,H,W
        image = tensor2image(frame_tensor)
        fifo_video_frames.append(image)

    save_file = os.path.join(save_dir,f"saved_video_final_long.mp4")
    export_to_video(fifo_video_frames, save_file, args.output_fps)

    return fifo_video_frames


