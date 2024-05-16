CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch --nproc_per_node=2 \
    --master_port=26666 train_scripts/train_internvid_open.py \
    configs/pixart_config/PixArt_xl2_img256_internvid.py \
    --work_dir output/debug \
    --load_from output/train_open_InternVId_256_500000images_resume/checkpoints/epoch_696_step_16549000.pth \