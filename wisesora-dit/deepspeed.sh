deepspeed --include localhost:4,5,6,7 \
  train_scripts/train_internvid_open_ds2.py \
  configs/pixart_config/PixArt_xl2_img256_internvid.py \
  --load_from output/train_open_InternVId_256_500000images_resume_1/checkpoints/epoch_2_step_19000.pth \
  --work_dir output/debug 