export NCCL_IB_DISABLE=1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 \
  --master_port=26662 \
  --nnodes=4 \
  --node_rank=3 \
  --master_addr="173.0.42.2" \
  train_scripts/train_internvid_open.py \
  configs/pixart_config/PixArt_xl2_img256_internvid.py \
  --load_from output/train_open_InternVId_256_500000images_resume_1/checkpoints/epoch_2_step_19000.pth \
  --work_dir output/train_open_InternVId_256_500000images_resume_1 \

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 \
#   --master_port=26662 \
#   --nnodes=2 \
#   --node_rank=1 \
#   --master_addr="173.0.42.2" \
#   train_scripts/train_internvid_open.py \
#   configs/pixart_config/PixArt_xl2_img256_internvid.py \
#   --load_from output/train_open_InternVId_256/checkpoints/epoch_580_step_414000.pth \
#   --work_dir output/train_open_InternVId_256 \

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 \
#   --master_port=26662 \
#   --nnodes=2 \
#   --node_rank=1 \
#   --master_addr="173.0.93.2" \
#   train_scripts/train_internvid_open.py \
#   configs/pixart_config/PixArt_xl2_img256_internvid.py \
#   --load_from output/train_open_InternVId_256/checkpoints/epoch_580_step_414000.pth \
#   --work_dir output/train_open_InternVId_256 \