import os
import torch
import argparse
import torchvision


from diffusers.schedulers import (DDIMScheduler, DDPMScheduler, PNDMScheduler, 
                                  EulerDiscreteScheduler, DPMSolverMultistepScheduler, 
                                  HeunDiscreteScheduler, EulerAncestralDiscreteScheduler,
                                  DEISMultistepScheduler, KDPM2AncestralDiscreteScheduler)
from diffusers.schedulers.scheduling_dpmsolver_singlestep import DPMSolverSinglestepScheduler
from diffusers.models import AutoencoderKL, AutoencoderKLTemporalDecoder
from omegaconf import OmegaConf
from transformers import T5EncoderModel, T5Tokenizer, AutoTokenizer
import os, sys
sys.path.append(os.path.split(sys.path[0])[0])
from tools.download import find_model
from diffusion.model.nets2.pipeline_videogen import VideoGenPipeline
from diffusion.model.nets2 import get_models,get_debug_models
import imageio
import math
from diffusion.model.t5 import T5Embedder


def save_video_grid(video, nrow=None):
    b, t, h, w, c = video.shape
    
    if nrow is None:
        nrow = math.ceil(math.sqrt(b))
    ncol = math.ceil(b / nrow)
    padding = 1
    video_grid = torch.zeros((t, (padding + h) * nrow + padding,
                           (padding + w) * ncol + padding, c), dtype=torch.uint8)
    
    for i in range(b):
        r = i // ncol
        c = i % ncol
        start_r = (padding + h) * r
        start_c = (padding + w) * c
        video_grid[:, start_r:start_r + h, start_c:start_c + w] = video[i]
    
    return video_grid


def main(args):
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # transformer_model = get_models(args).to(device, dtype=torch.float16)
    # state_dict = find_model(args.ckpt)
    # missing, unexpected = transformer_model.load_state_dict(state_dict['state_dict'], strict=False)

    # transformer_model = get_models(args).to(device, dtype=torch.float16) # 正式模型
    transformer_model = get_debug_models(args).to(device, dtype=torch.float16) # debug模型
    print(args.ckpt)
    state_dict = find_model(args.ckpt)
    # transformer_model.load_state_dict(state_dict['ema']) # 官方权重
    transformer_model.load_state_dict(state_dict['state_dict']) # 自己权重

    if args.enable_vae_temporal_decoder:
        vae = AutoencoderKLTemporalDecoder.from_pretrained(args.pretrained_model_path, subfolder="vae_temporal_decoder", torch_dtype=torch.float16).to(device)
        # vae = AutoencoderKL.from_pretrained("output/pretrained_models/sd-vae-ft-ema").to(device)
    else:
        # vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae", torch_dtype=torch.float16).to(device)
        vae = AutoencoderKL.from_pretrained("output/pretrained_models/sd-vae-ft-ema", torch_dtype=torch.float16).to(device)
    # tokenizer = T5Tokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
    # text_encoder = T5EncoderModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder", torch_dtype=torch.float16).to(device)
    t5 = T5Embedder(device="cuda", local_cache=True, cache_dir="output/pretrained_models/t5_ckpts", torch_dtype=torch.float16)
    tokenizer = t5.tokenizer
    text_encoder = t5.model
    # set eval mode
    transformer_model.eval()
    vae.eval()
    text_encoder.eval()

    if args.sample_method == 'DDIM':
        scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_path, 
                                                  subfolder="scheduler",
                                                  beta_start=args.beta_start, 
                                                  beta_end=args.beta_end, 
                                                  beta_schedule=args.beta_schedule,
                                                  variance_type=args.variance_type)
    elif args.sample_method == 'EulerDiscrete':
        scheduler = EulerDiscreteScheduler.from_pretrained(args.pretrained_model_path, 
                                                        subfolder="scheduler",
                                                        beta_start=args.beta_start, 
                                                        beta_end=args.beta_end, 
                                                        beta_schedule=args.beta_schedule,
                                                        variance_type=args.variance_type)
    elif args.sample_method == 'DDPM':
        scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_path, 
                                                  subfolder="scheduler",
                                                  beta_start=args.beta_start, 
                                                  beta_end=args.beta_end, 
                                                  beta_schedule=args.beta_schedule,
                                                  variance_type=args.variance_type)
    elif args.sample_method == 'DPMSolverMultistep':
        scheduler = DPMSolverMultistepScheduler.from_pretrained(args.pretrained_model_path, 
                                                  subfolder="scheduler",
                                                  beta_start=args.beta_start, 
                                                  beta_end=args.beta_end, 
                                                  beta_schedule=args.beta_schedule,
                                                  variance_type=args.variance_type)
    elif args.sample_method == 'DPMSolverSinglestep':
        scheduler = DPMSolverSinglestepScheduler.from_pretrained(args.pretrained_model_path, 
                                                  subfolder="scheduler",
                                                  beta_start=args.beta_start, 
                                                  beta_end=args.beta_end, 
                                                  beta_schedule=args.beta_schedule,
                                                  variance_type=args.variance_type)
    elif args.sample_method == 'PNDM':
        scheduler = PNDMScheduler.from_pretrained(args.pretrained_model_path, 
                                                  subfolder="scheduler",
                                                  beta_start=args.beta_start, 
                                                  beta_end=args.beta_end, 
                                                  beta_schedule=args.beta_schedule,
                                                  variance_type=args.variance_type)
    elif args.sample_method == 'HeunDiscrete':
        scheduler = HeunDiscreteScheduler.from_pretrained(args.pretrained_model_path, 
                                                  subfolder="scheduler",
                                                  beta_start=args.beta_start, 
                                                  beta_end=args.beta_end, 
                                                  beta_schedule=args.beta_schedule,
                                                  variance_type=args.variance_type)
    elif args.sample_method == 'EulerAncestralDiscrete':
        scheduler = EulerAncestralDiscreteScheduler.from_pretrained(args.pretrained_model_path, 
                                                  subfolder="scheduler",
                                                  beta_start=args.beta_start, 
                                                  beta_end=args.beta_end, 
                                                  beta_schedule=args.beta_schedule,
                                                  variance_type=args.variance_type)
    elif args.sample_method == 'DEISMultistep':
        scheduler = DEISMultistepScheduler.from_pretrained(args.pretrained_model_path, 
                                                  subfolder="scheduler",
                                                  beta_start=args.beta_start, 
                                                  beta_end=args.beta_end, 
                                                  beta_schedule=args.beta_schedule,
                                                  variance_type=args.variance_type)
    elif args.sample_method == 'KDPM2AncestralDiscrete':
        scheduler = KDPM2AncestralDiscreteScheduler.from_pretrained(args.pretrained_model_path, 
                                                  subfolder="scheduler",
                                                  beta_start=args.beta_start, 
                                                  beta_end=args.beta_end, 
                                                  beta_schedule=args.beta_schedule,
                                                  variance_type=args.variance_type)


    videogen_pipeline = VideoGenPipeline(vae=vae, 
                                 text_encoder=text_encoder, 
                                 tokenizer=tokenizer, 
                                 scheduler=scheduler, 
                                 transformer=transformer_model).to(device)
    # videogen_pipeline.enable_xformers_memory_efficient_attention()

    if not os.path.exists(args.save_img_path):
        os.makedirs(args.save_img_path)

    video_grids = []
    for prompt in args.text_prompt:
        print('Processing the ({}) prompt'.format(prompt))
        videos = videogen_pipeline(prompt, 
                                video_length=args.video_length, 
                                height=args.image_size[0], 
                                width=args.image_size[1], 
                                num_inference_steps=args.num_sampling_steps,
                                guidance_scale=args.guidance_scale,
                                enable_temporal_attentions=args.enable_temporal_attentions,
                                num_images_per_prompt=1,
                                mask_feature=True,
                                enable_vae_temporal_decoder=args.enable_vae_temporal_decoder
                                ).video
        try:
            imageio.mimwrite(args.save_img_path + prompt.replace(' ', '_') + '_%04d' % args.run_time + 'webv-imageio.mp4', videos[0], fps=8, quality=9,codec='libx264') # highest quality is 10, lowest is 0
        except:
            print('Error when saving {}'.format(prompt))
        video_grids.append(videos)
    video_grids = torch.cat(video_grids, dim=0)

    video_grids = save_video_grid(video_grids)

    torchvision.io.write_video(args.save_img_path + '_%04d' % args.run_time + '-.mp4', video_grids, fps=6)
    # imageio.mimwrite(args.save_img_path + '_%04d' % args.run_time + '-test.mp4', video_grids, fps=8, quality=5, codec='libx264')
    print('save path {}'.format(args.save_img_path))

    # save_videos_grid(video, f"./{prompt}.gif")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/wbv10m_train.yaml")
    args = parser.parse_args()

    main(OmegaConf.load(args.config))

