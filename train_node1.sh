CUDA_VISIBLE_DEVICES=6,7 accelerate launch --num_machines 2  
--num_processes 4 \
--machine_rank 1 \ 
--main_process_ip 173.0.42.2 \
--main_process_port 26662 \
train_scripts/train_internvid.py 
configs/pixart_config/PixArt_xl2_img256_internvid.py