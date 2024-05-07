CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=7 \
    --master_port=26662 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr="173.0.42.2" \
    train_scripts/train_internvid.py \
    configs/pixart_config/PixArt_xl2_img256_internvid.py \
    --load_from output/train_InternVId_FLT1_256/checkpoints/epoch_399_step_285285.pth \
    --work_dir "output/train_InternVId_FLT1_256" \