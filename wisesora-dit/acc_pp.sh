accelerate launch --config_file megatron_gpt_config.yaml --main_process_port 12321 --num_processes 4 \
  --mixed_precision "fp16" \
  train_scripts/train_internvid_open_pp.py \
  configs/pixart_config/PixArt_xl2_img256_internvid.py \
  --load_from output/train_open_InternVId_256_500000images_resume_1/checkpoints/epoch_2_step_19000.pth \
  --work_dir output/debug
