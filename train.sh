CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=7 \
    --master_port=26662 train_scripts/train_controlnet.py \
    configs/pixart_app_config/PixArt_xl2_img1024_controlHed_Half.py \
    --work-dir output/debug