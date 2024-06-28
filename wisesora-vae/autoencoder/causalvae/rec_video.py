import argparse
import cv2
import math
import numpy as np
import numpy.typing as npt
import os
import random
import torch
from typing import Optional
from PIL import Image
from decord import VideoReader, cpu
from torch.nn import functional as F
from pytorchvideo.transforms import ShortSideScale
from torchvision.transforms import Lambda, Compose

import sys
sys.path.append(".")

from models import getae_wrapper
from models.videovae import CausalVAEModel
from dataset.transform import CenterCropVideo, resize

def process_in_chunks(
    video_data: torch.Tensor,
    model: torch.nn.Module,
    chunk_size: int,
    overlap: int,
    device: str,
):
    assert (chunk_size + overlap - 1) % 4 == 0
    num_frames = video_data.size(2)
    output_chunks = []

    start = 0
    while start < num_frames:
        end = min(start + chunk_size, num_frames)
        if start + chunk_size + overlap < num_frames:
            end += overlap
        chunk = video_data[:, :, start:end, :, :]
        
        with torch.no_grad():
            chunk = chunk.to(device)
            latents = model.encode(chunk)
            recon_chunk = model.decode(latents.half()).cpu().float() # b t c h w
            recon_chunk = recon_chunk.permute(0, 2, 1, 3, 4)

        if output_chunks:
            overlap_step = min(overlap, recon_chunk.shape[2])
            overlap_tensor = (
                output_chunks[-1][:, :, -overlap_step:] * 1 / 4
                + recon_chunk[:, :, :overlap_step] * 3 / 4
            )
            output_chunks[-1] = torch.cat(
                (output_chunks[-1][:, :, :-overlap], overlap_tensor), dim=2
            )
            if end < num_frames:
                output_chunks.append(recon_chunk[:, :, overlap:])
            else:
                output_chunks.append(recon_chunk[:, :, :, :, :])
        else:
            output_chunks.append(recon_chunk)
        start += chunk_size
    return torch.cat(output_chunks, dim=2).permute(0, 2, 1, 3, 4)


def array_to_video(image_array: npt.NDArray, fps: float = 30.0, output_file: str = 'output_video.mp4') -> None:
    height, width, channels = image_array[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    video_writer = cv2.VideoWriter(output_file, fourcc, float(fps), (width, height))

    for image in image_array:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        video_writer.write(image_rgb)

    video_writer.release()


def custom_to_video(x: torch.Tensor, fps: float = 2.0, output_file: str = 'output_video.mp4') -> None:
    x = x.detach().cpu()
    x = torch.clamp(x, -1, 1)
    x = (x + 1) / 2
    x = x.permute(0, 2, 3, 1).numpy()
    x = (255 * x).astype(np.uint8)
    array_to_video(x, fps=fps, output_file=output_file)
    return


def read_video(video_path: str, num_frames: int, sample_rate: int) -> torch.Tensor:
    decord_vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(decord_vr)
    sample_frames_len = sample_rate * num_frames

    if total_frames > sample_frames_len:
        s = random.randint(0, total_frames - sample_frames_len - 1)
        s = 0
        e = s + sample_frames_len
        num_frames = num_frames
    else:
        s = 0
        e = total_frames
        num_frames = int(total_frames / sample_frames_len * num_frames)
        print(f'sample_frames_len {sample_frames_len}, only can sample {num_frames * sample_rate}', video_path,
              total_frames)

    frame_id_list = np.linspace(s, e - 1, num_frames, dtype=int)
    video_data = decord_vr.get_batch(frame_id_list).asnumpy()
    video_data = torch.from_numpy(video_data)
    video_data = video_data.permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)
    return video_data


class ResizeVideo:
    def __init__(
            self,
            size,
            aspect_ratio,
            keep_aspect,
            interpolation_mode="bilinear",
    ):
        self.size = size
        self.aspect_ratio = aspect_ratio
        self.keep_aspect = keep_aspect
        self.interpolation_mode = interpolation_mode

    def __call__(self, clip):
        _, _, h, w = clip.shape
        if not self.keep_aspect:
            if w < h:
                new_h = int(math.floor((float(h) / w) * self.size[1]))
                new_w = self.size[1]
            else:
                new_h = self.size[0]
                new_w = int(math.floor((float(w) / h) * self.size[0]))
        else:
            new_h = int(h * self.aspect_ratio)
            new_w = int(w * self.aspect_ratio)
        return torch.nn.functional.interpolate(
            clip, size=(new_h, new_w), mode=self.interpolation_mode, align_corners=False, antialias=True
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size}, interpolation_mode={self.interpolation_mode}"


def preprocess(video_data: torch.Tensor, height_width: list = [128, 256], crop_size: Optional[int] = None,
               aspect_ratio: float = 1.0, keep_aspect: bool = True) -> torch.Tensor:
    transform = Compose(
        [
            Lambda(lambda x: ((x / 255.0) * 2 - 1)),
            ResizeVideo(size=height_width, aspect_ratio=aspect_ratio, keep_aspect=keep_aspect),
            CenterCropVideo(crop_size) if crop_size is not None else Lambda(lambda x: x),
        ]
    )

    video_outputs = transform(video_data)
    video_outputs = torch.unsqueeze(video_outputs, 0)

    return video_outputs


def main(args: argparse.Namespace):
    device = args.device
    kwarg = {}
    vae = getae_wrapper(args.ae)(args.model_path, subfolder="vae", cache_dir='cache_dir', **kwarg).to(device)
    if args.enable_tiling:
        vae.vae.enable_tiling()
        vae.vae.tile_overlap_factor = args.tile_overlap_factor
    vae.eval()
    vae = vae.to(device)
    vae = vae.half()

    with torch.no_grad():
        x_vae = preprocess(read_video(args.video_path, args.num_frames, args.sample_rate), args.resolution,
                           args.crop_size, args.aspect_ratio, args.keep_aspect)
        x_vae = x_vae.to(device, dtype=torch.float16)  # b c t h w
        if args.enable_time_chunk:
            video_recon = process_in_chunks(x_vae, vae, 7, 2, device)
        else:
            latents = vae.encode(x_vae)
            latents = latents.to(torch.float16)
            video_recon = vae.decode(latents)  # b t c h w

    if video_recon.shape[2] == 1:
        x = video_recon[0, 0, :, :, :]
        x = x.squeeze()
        x = x.detach().cpu().numpy()
        x = np.clip(x, -1, 1)
        x = (x + 1) / 2
        x = (255 * x).astype(np.uint8)
        x = x.transpose(1, 2, 0)
        image = Image.fromarray(x)
        image.save(args.rec_path.replace('mp4', 'jpg'))
    else:
        custom_to_video(video_recon[0], fps=args.fps, output_file=args.rec_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, default='')
    parser.add_argument('--rec_path', type=str, default='')
    parser.add_argument('--ae', type=str, default='')
    parser.add_argument('--model_path', type=str, default='results/pretrained')
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--resolution', metavar='N', type=int, nargs='+')
    parser.add_argument('--crop_size', type=int, default=None)
    parser.add_argument('--num_frames', type=int, default=100)
    parser.add_argument('--sample_rate', type=int, default=1)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--tile_overlap_factor', type=float, default=0.25)
    parser.add_argument('--aspect_ratio', type=float, default=2.0)
    parser.add_argument('--keep_aspect', action='store_true')
    parser.add_argument('--enable_tiling', action='store_true')
    parser.add_argument('--enable_time_chunk', action='store_true')

    args = parser.parse_args()
    main(args)
