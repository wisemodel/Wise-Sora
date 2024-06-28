## 使用说明

### Latest Samples
我们在InternVid数据集上选取了部分数据，训练了7s 256×256，7s 512x512的模型，为了支持更高分辨率与更高时长视频的训练。我们做了以下优化：

对于256分辨率训练
1. Text encoder bf16 半精度推理——> 35帧到45帧
2. Text encoder，部分参数offload到cpu——>45帧到56帧，7s视频

为了支持更大分辨率的训练，我们开启了grad checkpointting,以时间换空间，最大可支持512 128帧的视频训练。

在未来的版本中，我们预计添加以下功能：

1. 高质量数据训练版本，我们预计从Pandas70M中筛选美学评分较高的视频，用于训练数据。

2. 启用FSDP与Xformers技术，以支持720P与1080P长视频的训练。

3. 高质量的vae技术

4. 高质量的视频打标

### 安装
```
# create a virtual env and activate (conda as an example)
conda create -n wisesora python=3.9
conda activate wisesora
# download the repo
git clone 
cd wisesora
pip install -r requirements.txt
```


### 模型权重
t5权重

vae权重

7s 512权重
"output/debug_r512_f32_56f_10000_grad/checkpoints/epoch_32_step_10000.pth"

7s 256权重
"output/net2_1000000/checkpoints/epoch_9_step_201200.pth"

### 推理
在t2v_sample1.yaml指定text_prompt, ckpt, pretrained_model_path, 然后执行执行以下命令

`python scripts/sample_t2v.py --config configs/transformer/t2v_sample1.yaml`

### 训练

单卡训练

`python -m torch.distributed.launch --nproc_per_node=1 --master_port=26665 train_scripts/train_internvid_nets2_fp32.py configs/pixart_config/PixArt_xl2_img256_internvid_debug.py --work_dir output/debug_r512_f32_18f_100`

单机8卡训练

`python -m torch.distributed.launch --nproc_per_node=8 --master_port=26665 train_scripts/train_internvid_nets2_fp32.py configs/pixart_config/PixArt_xl2_img256_internvid_debug.py --work_dir output/debug_r512_f32_18f_100`

多机多卡训练

例如 4机32卡

主节点执行

`python -m torch.distributed.launch --nproc_per_node=8 --master_port=26664 --node_rank=0 --nnodes=4 --master_addr=主节点地址 train_scripts/train_internvid_nets2_fp32.py  configs/pixart_config/PixArt_xl2_img256_internvid_debug.py --work_dir 输出文件目录  --load_from 权重地址

子节点1

`python -m torch.distributed.launch --nproc_per_node=8 --master_port=26664 --node_rank=1 --nnodes=4 --master_addr=主节点地址 train_scripts/train_internvid_nets2_fp32.py  configs/pixart_config/PixArt_xl2_img256_internvid_debug.py --work_dir 输出文件目录  --load_from 权重地址
`

子节点2

`python -m torch.distributed.launch --nproc_per_node=8 --master_port=26664 --node_rank=2 --nnodes=4 --master_addr=主节点地址 train_scripts/train_internvid_nets2_fp32.py  configs/pixart_config/PixArt_xl2_img256_internvid_debug.py --work_dir 输出文件目录  --load_from 权重地址
`

子节点3

`python -m torch.distributed.launch --nproc_per_node=8 --master_port=26664 --node_rank=3 --nnodes=4 --master_addr=主节点地址 train_scripts/train_internvid_nets2_fp32.py  configs/pixart_config/PixArt_xl2_img256_internvid_debug.py --work_dir 输出文件目录  --load_from 权重地址
`

### Acknowledgement
我们列出了我们参考的项目，我们非常感激他们在开源方面杰出的工作和慷慨的贡献。

[Open-Sora](https://github.com/hpcaitech/Open-Sora): Democratizing Efficient Video Production for All

[Latte](https://github.com/Vchitect/Latte): Latent Diffusion Transformer for Video Generation

[PixArt-α](https://github.com/PixArt-alpha/PixArt-alpha): Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis