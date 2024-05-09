# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------
import math
import torch
import torch.nn as nn
import os
import numpy as np
from einops import rearrange, repeat
from timm.models.layers import DropPath
from timm.models.vision_transformer import PatchEmbed, Mlp

import sys
sys.path.append('/home/lijunjie/work/PixArt-alpha/diffusion')

from diffusion.model.builder import MODELS
from diffusion.model.utils import auto_grad_checkpoint, to_2tuple
from diffusion.model.nets.PixArt_blocks import t2i_modulate, CaptionEmbedder, WindowAttention, MultiHeadCrossAttention, T2IFinalLayer, TimestepEmbedder, LabelEmbedder, FinalLayer
from diffusion.utils.logger import get_root_logger


class PixArtBlock(nn.Module):
    """
    A PixArt block with adaptive layer norm (adaLN-single) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, drop_path=0., window_size=0, input_size=None, use_rel_pos=False, **block_kwargs):
        super(PixArtBlock, self).__init__()
        self.hidden_size = hidden_size
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = WindowAttention(hidden_size, num_heads=num_heads, qkv_bias=True,
                                    input_size=input_size if window_size == 0 else (window_size, window_size),
                                    use_rel_pos=use_rel_pos, **block_kwargs)
        self.cross_attn = MultiHeadCrossAttention(hidden_size, num_heads, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # to be compatible with lower version pytorch
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=approx_gelu, drop=0)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.window_size = window_size
        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size ** 0.5)

    def forward(self, x, y, t, mask=None, **kwargs):
        B, N, C = x.shape
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.scale_shift_table[None] + t.reshape(B, 6, -1)).chunk(6, dim=1)
        x = x + self.drop_path(gate_msa * self.attn(t2i_modulate(self.norm1(x), shift_msa, scale_msa)).reshape(B, N, C))
        x = x + self.cross_attn(x, y, mask)
        x = x + self.drop_path(gate_mlp * self.mlp(t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)))

        return x


class PixArtTempBlock(nn.Module):
    """
    A PixArt block with adaptive layer norm (adaLN-single) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, drop_path=0., window_size=0, input_size=None, use_rel_pos=False, **block_kwargs):
        super(PixArtTempBlock, self).__init__()
        self.hidden_size = hidden_size
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = WindowAttention(hidden_size, num_heads=num_heads, qkv_bias=True,
                                    input_size=input_size if window_size == 0 else (window_size, window_size),
                                    use_rel_pos=use_rel_pos, **block_kwargs)
        self.cross_attn = MultiHeadCrossAttention(hidden_size, num_heads, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # to be compatible with lower version pytorch
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=approx_gelu, drop=0)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.window_size = window_size
        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size ** 0.5)

    def forward(self, x, y, t, temp_embed, mask=None, **kwargs):
        B, N, C = x.shape
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.scale_shift_table[None] + t.reshape(B, 6, -1)).chunk(6, dim=1)
        x1 = x + self.drop_path(gate_msa * self.attn(t2i_modulate(self.norm1(x), shift_msa, scale_msa)).reshape(B, N, C))
        x1 = x1 + temp_embed
        x1 = x1 + self.cross_attn(x1, y, mask)
        x1 = self.drop_path(gate_mlp * self.mlp(t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)))
        x = x + x1

        return x

#############################################################################
#                                 Core PixArt Model                                #
#################################################################################
@MODELS.register_module()
class PixArtT2V(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(self, input_size=32, patch_size=2, in_channels=4, hidden_size=1152, depth=28, num_heads=16, mlp_ratio=4.0, class_dropout_prob=0.1, pred_sigma=True, drop_path: float = 0., window_size=0, window_block_indexes=None, use_rel_pos=False, caption_channels=4096, lewei_scale=1.0, config=None, model_max_length=120, num_frames = 16, **kwargs):
        if window_block_indexes is None:
            window_block_indexes = []
        super(PixArtT2V, self).__init__()
        self.pred_sigma = pred_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if pred_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.lewei_scale = lewei_scale,

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        num_patches = self.x_embedder.num_patches
        self.base_size = input_size // self.patch_size
        # Will use fixed sin-cos embedding:
        self.register_buffer("pos_embed", torch.zeros(1, num_patches, hidden_size))
        self.temp_embed = nn.Parameter(torch.zeros(1, num_frames, hidden_size), requires_grad=True)

        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.t_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        self.y_embedder = CaptionEmbedder(in_channels=caption_channels, hidden_size=hidden_size, uncond_prob=class_dropout_prob, act_layer=approx_gelu, token_num=model_max_length)
        drop_path = [x.item() for x in torch.linspace(0, drop_path, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            PixArtBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, drop_path=drop_path[i],
                          input_size=(input_size // patch_size, input_size // patch_size),
                          window_size=window_size if i in window_block_indexes else 0,
                          use_rel_pos=use_rel_pos if i in window_block_indexes else False)
            for i in range(depth)
        ])

        self.temp_blocks = nn.ModuleList([
            PixArtTempBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, drop_path=drop_path[i],
                          input_size=(input_size // patch_size, input_size // patch_size),
                          window_size=window_size if i in window_block_indexes else 0,
                          use_rel_pos=use_rel_pos if i in window_block_indexes else False)
            for i in range(depth)
        ])

        self.final_layer = T2IFinalLayer(hidden_size, patch_size, self.out_channels)

        self.initialize_weights()

        if config:
            logger = get_root_logger(os.path.join(config.work_dir, 'train_log.log'))
            logger.warning(f"lewei scale: {self.lewei_scale}, base size: {self.base_size}")
        else:
            print(f'Warning: lewei scale: {self.lewei_scale}, base size: {self.base_size}')

    def forward(self, x, timestep, y, mask=None, data_info=None, use_image_num=4, y_image=None, **kwargs):
        """
        Forward pass of PixArt.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N, 1, 120, C) tensor of class labels
        """
        # 输入为视频 假如输入X为（1，20，4，64，64）timestep=1 y=[1,1,120,4096] use_image_num=4
        # 以下分析数据流情况
        # self.training = True
        batches, frames, channels, high, weight = x.shape 
        x = rearrange(x, 'b f c h w -> (b f) c h w')  # x: torch.Size([20, 4, 64, 64])
        # print("x rearrange shape:",x.shape)  
        x = x.to(self.dtype)
        timestep = timestep.to(self.dtype)
        y = y.to(self.dtype)
        pos_embed = self.pos_embed.to(self.dtype) # pos_embed:  torch.Size([1, 1024, 1152])
        self.h, self.w = x.shape[-2]//self.patch_size, x.shape[-1]//self.patch_size
        x = self.x_embedder(x) + pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        # print("x embedding",x.shape)        # x: torch.Size([20, 1024, 1152])
        t = self.t_embedder(timestep.to(x.dtype))  # (N, D) 
        # print("t embedding",t.shape)        # t: torch.Size([1, 1152])
        t0 = self.t_block(t)                # (N, D*6) 便于后面chunk为6份
        # print("t0 shape", t0.shape)         # t0: torch.Size([1, 6912])
        y = self.y_embedder(y, self.training)  # (N, 1, L, D)
        # print("y embedding", y.shape)       # y: torch.Size([1, 1, 120, 1152])
        # print("self.temp_embed.shape", self.temp_embed.shape)
        # print("self.pos_embed.shape", self.pos_embed.shape)
        timestep_spatial = repeat(t0, 'n d -> (n c) d', c=self.temp_embed.shape[1] + use_image_num)  # self.temp_embed.shape[1] + use_image_num == frames
        # print("timestep_spatial", timestep_spatial.shape)  # timestep_spatial: torch.Size([20, 6912])
        timestep_temp = repeat(t0, 'n d -> (n c) d', c=self.pos_embed.shape[1]) # timestep_temp: torch.Size([1024, 6912])
        # print("timestep_temp", timestep_temp.shape)
        # print("self.training", self.training)
        if self.training:
            if y_image is not None:
                all_y_image_emb = []
                for y_image_batch in y_image:
                    y_image_emb = []
                    for y_image_single in y_image_batch:
                        y_image_emb.append(self.y_embedder(y_image_single, self.training))
                    y_image_emb = torch.cat(y_image_emb, dim=0) 
                    all_y_image_emb.append(y_image_emb)   
                all_y_image_emb = torch.cat(all_y_image_emb, dim=0) 
                if batches == 1:
                    all_y_image_emb = all_y_image_emb[None,:]
                y_spatial = repeat(y, 'n a b d -> n c a b d', c=self.temp_embed.shape[1])
                y_spatial = torch.cat([y_spatial, all_y_image_emb], dim=1)
                y_spatial = rearrange(y_spatial, 'n c a b d -> (n c) a b d')
            else:
                y_spatial = repeat(y, 'n a b d -> (n c) a b d', c=self.temp_embed.shape[1] + use_image_num)
                y_spatial_mask = repeat(mask, 'n a -> (n c) a', c=self.temp_embed.shape[1] + use_image_num)
        else:
            y_spatial = repeat(y, 'n a b d -> (n c) a b d', c=self.temp_embed.shape[1] + use_image_num) 
            y_spatial_mask = repeat(mask, 'n a -> (n c) a', c=self.temp_embed.shape[1] + use_image_num)
        # print("y_spatial", y_spatial.shape) # y_spatial: torch.Size([20, 1, 120, 1152])
        y_temp = repeat(y, 'n a b d -> (n c) a b d', c=self.pos_embed.shape[1])
        y_temp_mask = repeat(mask, 'n a -> (n c) a', c=self.pos_embed.shape[1])
        # print("y_temp", y_temp.shape)      # y_temp torch.Size([1024, 1, 120, 1152])
        if mask is not None:
            if mask.shape[0] != y.shape[0]:
                mask = mask.repeat(y.shape[0] // mask.shape[0], 1)
                y_spatial_mask = y_spatial_mask.repeat(y_spatial.shape[0] // y_spatial_mask.shape[0], 1)
                y_temp_mask = y_temp_mask.repeat(y_temp.shape[0] // y_temp_mask.shape[0], 1)
            mask = mask.squeeze(1).squeeze(1)
            y = y.squeeze(1).masked_select(mask.unsqueeze(-1) != 0).view(1, -1, x.shape[-1])
            y_lens = mask.sum(dim=1).tolist()
            y_spatial_mask = y_spatial_mask.squeeze(1).squeeze(1)
            y_spatial = y_spatial.squeeze(1).masked_select(y_spatial_mask.unsqueeze(-1) != 0).view(1, -1, x.shape[-1])
            y_spatial_lens = y_spatial_mask.sum(dim=1).tolist()
            y_temp_mask = y_temp_mask.squeeze(1).squeeze(1)
            y_temp = y_temp.squeeze(1).masked_select(y_temp_mask.unsqueeze(-1) != 0).view(1, -1, x.shape[-1])
            y_temp_lens = y_temp_mask.sum(dim=1).tolist()
        else:
            y_lens = [y.shape[2]] * y.shape[0]
            y = y.squeeze(1).view(1, -1, x.shape[-1])
            y_spatial_lens = [y_spatial.shape[2]] * y_spatial.shape[0]
            y_spatial = y_spatial.squeeze(1).reshape(1, -1, x.shape[-1])
            y_temp_lens = [y_temp.shape[2]] * y_temp.shape[0]
            y_temp = y_temp.squeeze(1).reshape(1, -1, x.shape[-1])
        # for i in range(0, len(self.blocks), 2):
            # spatial_block, temp_block = self.blocks[i:i+2]
        for i, (spatial_block, temp_block) in enumerate(zip(self.blocks, self.temp_blocks)):
            # print("spatial_block",x.shape,y_spatial.shape,timestep_spatial.shape)
            # x = auto_grad_checkpoint(spatial_block, x, y_spatial, timestep_spatial, y_spatial_lens)  # (N, T, D) #support grad checkpoint
            x = spatial_block(x, y_spatial, timestep_spatial, y_spatial_lens)
            x = rearrange(x, '(b f) t d -> (b t) f d', b=batches)
            x_video = x[:, :(frames-use_image_num), :]
            # if i==0:
            #     x_video = x_video + self.temp_embed 
            x_image = x[:, (frames-use_image_num):, :]
            # x_video = auto_grad_checkpoint(temp_block, x_video, y_temp, timestep_temp, self.temp_embed, y_temp_lens)  # (N, T, D) #support grad checkpoint
            x_video = temp_block(x_video, y_temp, timestep_temp, self.temp_embed, y_temp_lens)
            x = torch.cat([x_video, x_image], dim=1)
            x = rearrange(x, '(b t) f d -> (b f) t d', b=batches)
        t = repeat(t,'b t-> (b f) t',f = frames)
        x = self.final_layer(x, t)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        x = rearrange(x, '(b f) c h w -> b f c h w', b = batches)
        return x

    def forward_with_dpmsolver(self, x, timestep, y, mask=None, **kwargs):
        """
        dpm solver donnot need variance prediction
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        model_out = self.forward(x, timestep, y, mask)
        if model_out.ndim == 5:
            model_out = rearrange(model_out,'b f c h w -> (b f) c h w')
        return model_out.chunk(2, dim=1)[0]

    def forward_with_cfg(self, x, timestep, y, cfg_scale, mask=None, **kwargs):
        """
        Forward pass of PixArt, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, timestep, y, mask, kwargs)
        model_out = model_out['x'] if isinstance(model_out, dict) else model_out
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        return x.reshape(shape=(x.shape[0], c, h * p, h * p))

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5), lewei_scale=self.lewei_scale, base_size=self.base_size)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        temp_embed = get_1d_sincos_temp_embed(self.temp_embed.shape[-1], self.temp_embed.shape[-2])
        self.temp_embed.data.copy_(torch.from_numpy(temp_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.t_block[1].weight, std=0.02)

        # Initialize caption embedding MLP:
        nn.init.normal_(self.y_embedder.y_proj.fc1.weight, std=0.02)
        nn.init.normal_(self.y_embedder.y_proj.fc2.weight, std=0.02)

        # Zero-out adaLN modulation layers in PixArt blocks:
        for block in self.blocks:
            nn.init.constant_(block.cross_attn.proj.weight, 0)
            nn.init.constant_(block.cross_attn.proj.bias, 0)

        for block in self.temp_blocks:           
            nn.init.constant_(block.mlp.fc2.weight, 0)
            nn.init.constant_(block.mlp.fc2.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    @property
    def dtype(self):
        return next(self.parameters()).dtype


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0, lewei_scale=1.0, base_size=16):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    if isinstance(grid_size, int):
        grid_size = to_2tuple(grid_size)
    grid_h = np.arange(grid_size[0], dtype=np.float32) / (grid_size[0]/base_size) / lewei_scale
    grid_w = np.arange(grid_size[1], dtype=np.float32) / (grid_size[1]/base_size) / lewei_scale
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size[1], grid_size[0]])

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    return np.concatenate([emb_h, emb_w], axis=1)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    return np.concatenate([emb_sin, emb_cos], axis=1)


def get_1d_sincos_temp_embed(embed_dim, length):
    pos = torch.arange(0, length).unsqueeze(1)
    return get_1d_sincos_pos_embed_from_grid(embed_dim, pos)

#################################################################################
#                                   PixArt Configs                                  #
#################################################################################
@MODELS.register_module()
def PixArt_XL_2_T2V(**kwargs):
    return PixArtT2V(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)