#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="0"

python -m torch.distributed.run \
    --nproc_per_node 1 \
    --nnodes 1 \
    --master_port 23456 \
    rec_video.py \
    --model_path /path/to/directory/of/weights \
    --video_path ../../assets/origin_video_0.mp4 \
    --rec_path results/rec_video_0_release_v1_keep_aspect_x2.mp4 \
    --ae CausalVAEModel_4x8x8 \
    --device cuda \
    --fps 30 \
    --sample_rate 1 \
    --num_frames 300 \
    --resolution 1080 1920 \
    --aspect_ratio 2.0 \
    --keep_aspect \
    --enable_tiling \
    --enable_time_chunk
