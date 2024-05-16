import os
import json
import numpy as np
import torch
from decord import VideoReader
from decord import gpu,cpu
from torch.utils.data import Dataset
import sys
sys.path.append('/home/lijunjie/work/PixArt-alpha')
from diffusion.data.builder import DATASETS
from torchvision.transforms.functional import to_pil_image, to_tensor
import pandas as pd


@DATASETS.register_module()
class InternVId(Dataset):
    def __init__(self,
                 root,
                 transform=None,
                 resolution=256,
                 mask_ratio=0.0,
                 mask_type='null',
                 **kwargs):
        self.root = root
        self.transform = transform
        self.mask_type = mask_type
        self.mask_ratio = mask_ratio
        self.resolution = resolution
        self.video_samples = []
        self.caption_samples = []
        self.txt_feat_samples = []
        self.vae_feat_samples = []
        json_path = "/home/lijunjie/work/datasets/InternVId-FLT/InternVId-FLT_1.jsonl"
        # 读取 JSON 文件
        df = pd.read_json(json_path, lines=True)

        # 取前2000个
        df = df.head(500000)
        # df = df.iloc[48037:48038]

        print(df)

        # 构造视频文件名和路径
        df['video_name'] = df['YoutubeID'].astype(str) + "_" + df['Start_timestamp'].astype(str) + "_" + df['End_timestamp'].astype(str) + ".mp4"
        df['video_path'] = df['video_name'].apply(lambda x: os.path.join(self.root, x))

        print(df['video_path'])

        # 过滤存在的视频文件
        df['exists'] = df['video_path'].apply(os.path.exists)
        df = df[df['exists']]

        # print(df)

        # 赋值给类属性
        self.video_samples = df['video_path'].tolist()
        self.caption_samples = df['Caption'].tolist()
        self.ori_imgs_nums = len(df)

    def getdata(self, idx, frame_interval=4, needed_frames=16):
        video_path = self.video_samples[idx]
        data_info = {'img_hw': torch.tensor([self.resolution, self.resolution], dtype=torch.float32),
                     'aspect_ratio': torch.tensor(1.)}
        if not os.path.exists(video_path):
            # print(f"Warning: File {video_path} not found or is not a file.")
            return None 
        try:
            video = VideoReader(video_path,ctx=cpu())
        except:
             return None
        total_frames = len(video)  # 获取视频总帧数
        max_frame_index = frame_interval * (needed_frames - 1)
        if total_frames < max_frame_index+1:
            # print(f"Video {video_path} does not have enough frames (needed 16, got {total_frames}).")
            return None
        import random
        start_frame = random.randint(0, total_frames - max_frame_index - 1)  # 随机选择起始帧
        selected_frames = range(start_frame, start_frame + max_frame_index + 1, frame_interval)  # 根据间隔选取帧
        frames = video.get_batch(list(selected_frames) )# 获取连续的needed_frames帧
        video = torch.tensor(frames.asnumpy()).clone()
        video = torch.permute(video, (0, 3, 1, 2)).clone()

        transformed_frames = []
        if self.transform:
            for frame in video:
                pil_image = to_pil_image(frame)  # 转换为 PIL 图像
                transformed_image = self.transform(pil_image)  # 应用 PIL 图像的转换
                transformed_frames.append(transformed_image)
            video = torch.stack(transformed_frames)
        # print(f"Video tensor shape after transformations: {video.shape}")  # Debugging
        video_caption = self.caption_samples[idx]
        data_info["mask_type"] = self.mask_type
        attention_mask = None

        return video, video_caption, attention_mask, data_info

    def __getitem__(self, idx):
        data = self.getdata(idx)
        if data is not None:
            return data


    def __len__(self):
        return len(self.video_samples)



# if __name__ == "__main__":
#     # 查看InterVid json的内容
#     import os
#     from decord import VideoReader
#     from decord import gpu,cpu
#     DATA_ROOT = "/home/lijunjie/work/datasets/InternVId-FLT/InternVId-FLT_1"
#     import json
#     json_path = "/home/lijunjie/work/datasets/InternVId-FLT/InternVid-10M-flt.jsonl"
#     with open(json_path, 'r') as file:
#         for line in file:
#             data = json.loads(line)
#             video_name = data['YoutubeID'] + "_" + data['Start_timestamp'] + "_" + data["End_timestamp"] + ".mp4"
#             video_path = os.path.join(DATA_ROOT, video_name)
#             # video_path = "/home/lijunjie/work/datasets/InternVId-FLT/InternVId-FLT_1/0TDAwrav8Dc_00:01:51.578_00:01:53.647.mp4"
#             try:
#                 vr = VideoReader(video_path, ctx=cpu())
#             except FileNotFoundError:
#                 print(f"File not found: {video_path}")
#                 continue
#             except Exception as e:
#                 print(f"An error occurred: {e}")
#                 continue
#             frames = vr.get_batch(range(len(vr)))
#             breakpoint()
    