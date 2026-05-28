from argparse import ArgumentParser
from omegaconf import OmegaConf
import os
import torch
import numpy as np
from PIL import Image
import PIL
import imageio
import yaml

from pytorch_lightning import seed_everything

from scripts.evaluation.funcs import *
from utils.utils import instantiate_from_config
from lvdm.models.samplers.ddim import DDIMSampler
import json


def _load_miga_config(config_path):
    """Load MIGA hyperparameters from YAML config file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config.get('miga', {})


def export_to_video(video_frames, output_video_path, fps):
    if isinstance(video_frames[0], np.ndarray):
        if video_frames[0].dtype != np.uint8:
            video_frames = [(frame * 255).astype(np.uint8) for frame in video_frames]

    elif isinstance(video_frames[0], PIL.Image.Image):
        video_frames = [np.array(frame) for frame in video_frames]

    with imageio.get_writer(output_video_path, fps=fps) as writer:
        for frame in video_frames:
            writer.append_data(frame)



def main(args):
    ## step 1: model config
    ## -----------------------------------------------------------------
    config = OmegaConf.load(args.config)
    #data_config = config.pop("data", OmegaConf.create())
    model_config = config.pop("model", OmegaConf.create())
    model = instantiate_from_config(model_config)
    model = model.cuda()
    assert os.path.exists(args.ckpt_path), f"Error: checkpoint [{args.ckpt_path}] Not Found!"
    model = load_model_checkpoint(model, args.ckpt_path)
    model.eval()


    ## sample shape
    assert (args.height % 16 == 0) and (args.width % 16 == 0), "Error: image size [h,w] should be multiples of 16!"
    ## latent noise shape
    h, w = args.height // 8, args.width // 8   # 40, 64
    frames = args.video_length          # 16
    channels = model.channels

    ## step 2: load data
    ## -----------------------------------------------------------------
    save_dir = f"{args.save_dir}/{args.exp_name}"
    os.makedirs(save_dir, exist_ok=True)
    prompt = args.prompt

    ## step 3: run over samples
    ## -----------------------------------------------------------------
    batch_size = 1
    noise_shape = [batch_size, channels, frames, h, w]
    fps = torch.tensor([args.fps]*batch_size).to(model.device).long()

    text_emb = model.get_learned_conditioning([prompt])
    cond = {"c_crossattn": [text_emb], "fps": fps}
    ##################### initial inference for short video #####################
    base_tensor, ddim_sampler, _ = base_ddim_sampling(model, cond, noise_shape, \
                                        args.num_inference_steps, args.eta, args.unconditional_guidance_scale, \
                                        latents_dir=save_dir)
    save_gif(base_tensor, save_dir, "origin")

    ##################### MIGA inference for long video  #####################
    video_frames = generate_by_miga(
        args, model, cond, noise_shape, ddim_sampler, args.unconditional_guidance_scale, save_dir=save_dir, long_generate_setting=args
    )  # (512, 320) PIL.image

    print("finish!!! ")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default=None, help="Path to VideoCrafter2 checkpoint (model.ckpt)")
    parser.add_argument("--config", type=str, default="configs/inference_t2v_512_v2.0.yaml", help="VideoCrafter2 model config (yaml) path")
    parser.add_argument("--seed", type=int, default=321)
    parser.add_argument("--video_length", type=int, default=16, help="Number of latent frames f0")
    parser.add_argument("--num_partitions", "-n", type=int, default=4, help="n in paper")
    parser.add_argument("--num_inference_steps", type=int, default=64, help="Number of inference steps")
    parser.add_argument("--num_processes", type=int, default=1, help="Number of processes")
    parser.add_argument("--rank", type=int, default=0, help="Rank of the process (0~num_processes-1)")
    parser.add_argument("--height", type=int, default=320, help="Height of the output video")
    parser.add_argument("--width", type=int, default=512, help="Width of the output video")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--unconditional_guidance_scale", type=float, default=12.0, help="Classifier-free guidance scale")
    parser.add_argument("--eta", "-e", type=float, default=1.0)
    parser.add_argument("--use_mp4", action="store_true", default=True, help="Use mp4 format for output video")
    parser.add_argument("--output_fps", type=int, default=10, help="FPS of the output video")

    ############# MIGA args ##################
    parser.add_argument("--miga_config", type=str, default="../configs/videocrafter2.yaml",
                        help="Path to MIGA YAML configuration file.")
    parser.add_argument("--prompt", type=str, default=None,
                        help="The text prompt to generate the video from.")
    parser.add_argument("--save_dir", type=str, default="./outputs",
                        help="Directory to save generated videos.")
    parser.add_argument("--exp_name", type=str, default="demo_test",
                        help="Experiment name for saving results.")
    # The following MIGA args can override YAML config values
    parser.add_argument("--sampling_steps", type=int, default=None,
                        help="Total denoising steps T. (overrides YAML config)")
    parser.add_argument("--saw_width", type=int, default=None,
                        help="Zigzag width Lzig in Stage 1. (overrides YAML config)")
    parser.add_argument("--long_iter_nums", type=int, default=None,
                        help="Number of generated latent chunks. (overrides YAML config)")
    parser.add_argument("--temporal_memory_len", type=int, default=None,
                        help="Number of long-range guidance frames m_guid. (overrides YAML config)")
    parser.add_argument("--involve_resample", type=lambda x: x.lower() == 'true', default=None,
                        help="Whether to enable self-reflection. (overrides YAML config)")
    parser.add_argument("--resample_threshold", type=float, default=None,
                        help="Self-reflection threshold delta_adju. (overrides YAML config)")
    parser.add_argument("--save_inter_results", type=lambda x: x.lower() == 'true', default=None,
                        help="Whether to save intermediate results. (overrides YAML config)")

    args = parser.parse_args()

    # Load MIGA config from YAML and apply defaults (CLI overrides YAML)
    miga_cfg = _load_miga_config(args.miga_config)
    if args.sampling_steps is None:
        args.sampling_steps = miga_cfg.get('sampling_steps', 64)
    if args.saw_width is None:
        args.saw_width = miga_cfg.get('saw_width', 4)
    if args.long_iter_nums is None:
        args.long_iter_nums = miga_cfg.get('long_iter_nums', 30)
    if args.temporal_memory_len is None:
        args.temporal_memory_len = miga_cfg.get('temporal_memory_len', 4)
    if args.involve_resample is None:
        args.involve_resample = miga_cfg.get('involve_resample', False)
    if args.resample_threshold is None:
        args.resample_threshold = miga_cfg.get('resample_threshold', 0.05)
    if args.save_inter_results is None:
        args.save_inter_results = miga_cfg.get('save_inter_results', True)

    seed_everything(args.seed)
    main(args)
