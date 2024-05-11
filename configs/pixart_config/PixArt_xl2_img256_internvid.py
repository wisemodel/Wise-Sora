_base_ = ['../PixArt_xl2_internvid.py']
data_root = '/home/lijunjie/work/datasets/InternVId-FLT/InternVId-FLT_1'
data = dict(type='InternVId', root=data_root, transform='default_train')
image_size = 256

# model setting
window_block_indexes=[]
window_size=0
use_rel_pos=False
# model = 'PixArt_XL_2_T2V'
model = 'PixArtT2V_XL_2'
fp32_attention = True
# load_from = "output/pretrained_models/PixArt-XL-2-SAM-256x256.pth"
load_from = "output/train_InternVId_1_256/checkpoints/epoch_10000_step_10000.pth"
vae_pretrained = "output/pretrained_models/sd-vae-ft-ema"

# training setting
use_fsdp=False   # if use FSDP mode
num_workers=32
# num_workers=0
train_batch_size = 1 # 32
num_epochs = 25000 # 3
gradient_accumulation_steps = 1
grad_checkpointing = True
gradient_clip = 0.01
optimizer = dict(type='AdamW', lr=2e-5, weight_decay=3e-2, eps=1e-10)
lr_schedule_args = dict(num_warmup_steps=1000)
mixed_precision = 'bf16'
eval_sampling_steps = 200
log_interval = 20
save_model_epochs=5000
save_model_steps=1000
work_dir = 'output/train_InternVId_FLT1_256'
resume_from = dict(checkpoint="output/train_open_InternVId_256/checkpoints/epoch_580_step_414000.pth", load_ema=False, resume_optimizer=True, resume_lr_scheduler=True)
# resume_from = dict(checkpoint="output/train_InternVId_1_256/checkpoints/epoch_10000_step_10000.pth", load_ema=False, resume_optimizer=True, resume_lr_scheduler=True)
