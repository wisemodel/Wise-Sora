import argparse
import datetime
import os
import sys
import time
import types
import warnings
from copy import deepcopy
from pathlib import Path
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
from einops import rearrange, repeat
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import DistributedType
from accelerate.utils import DistributedDataParallelKwargs
from diffusers.models import AutoencoderKL
from mmcv.runner import LogBuffer
from torch.utils.data import RandomSampler
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
target_dir = os.path.join(current_dir, '..', '..', 'wisesora')
sys.path.append(target_dir)
from diffusion import IDDPM
from diffusion.data.builder import build_dataset, build_dataloader, set_data_root
from diffusion.model.builder import build_model
from diffusion.utils.checkpoint import save_checkpoint, load_checkpoint, load_checkpoint_net2
from diffusion.utils.data_sampler import AspectRatioBatchSampler, BalancedAspectRatioBatchSampler
from diffusion.utils.dist_utils import get_world_size, clip_grad_norm_
from diffusion.utils.logger import get_root_logger
from diffusion.utils.lr_scheduler import build_lr_scheduler
from diffusion.utils.misc import set_random_seed, read_config, init_random_seed, DebugUnderflowOverflow
from diffusion.utils.optimizer import build_optimizer, auto_scale_lr
from diffusion.model.t5 import T5Embedder
from diffusion.data.datasets import ASPECT_RATIO_256_TEST,ASPECT_RATIO_512_TEST
from diffusion.model.utils import prepare_prompt_ar
from diffusion.model.nets.pixart_t2v import PixArtT2V_XL_2
from diffusion.model.nets2 import get_models,get_debug_models
from diffusion.utils.dist_utils import synchronize
from diffusers.utils.import_utils import is_xformers_available


warnings.filterwarnings("ignore")  # ignore warning

current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))


def set_fsdp_env():
    os.environ["ACCELERATE_USE_FSDP"] = 'true'
    # os.environ["FSDP_AUTO_WRAP_POLICY"] = 'TRANSFORMER_BASED_WRAP'
    # os.environ["FSDP_AUTO_WRAP_POLICY"] = 'SIZE_BASED_WRAP'
    # os.environ["FSDP_MIN_NUM_PARAMS"] = '20000'
    os.environ["FSDP_BACKWARD_PREFETCH"] = 'BACKWARD_PRE'
    # os.environ["FSDP_TRANSFORMER_CLS_TO_WRAP"] = 'BasicTransformerBlock'
    # os.environ["FSDP_SHARDING_STRATEGY"] = 'FULL_SHARD'
    # os.environ["FSDP_STATE_DICT_TYPE"] = "SHARDED_STATE_DICT"
    # os.environ["FSDP_OFFLOAD_PARAMS"] = 'true'


def ema_update(model_dest: nn.Module, model_src: nn.Module, rate):
    param_dict_src = dict(model_src.named_parameters())
    for p_name, p_dest in model_dest.named_parameters():
        p_src = param_dict_src[p_name]
        assert p_src is not p_dest
        p_dest.data.mul_(rate).add_((1 - rate) * p_src.data)

def safe_collate(batch):
    batch = [item for item in batch if item is not None]
    videos = []
    captions = []
    attention_masks = []
    data_infos = []
    for item in batch:
        if item[0] is not None:  
            videos.append(item[0])
            captions.append(item[1])  
            if item[2] is not None:  
                attention_masks.append(item[2])
            else:
                attention_masks.append(torch.zeros_like(videos[-1][0], dtype=torch.float32))  
            data_infos.append(item[3])
    videos = torch.stack(videos) if videos else None
    return videos, captions, attention_masks, data_infos

def low_mem_vae(vae, model_input, micro_bs=16):
    with torch.no_grad():
        # Map input images to latent space + normalize latents:
        b, _, _, _, _ = model_input.shape
        chunk_outs = []
        model_input = rearrange(model_input, 'b f c h w -> (b f) c h w').contiguous()
        chunks = model_input.chunk(micro_bs, 0)
        for chunk in chunks:
            chunk_out = vae.encode(chunk).latent_dist.sample().mul_(vae.config.scaling_factor)
            chunk_outs.append(chunk_out)

        model_input = torch.cat(chunk_outs, dim=0)
        model_input = rearrange(model_input, '(b f) c h w -> b c f h w', b=b).contiguous() # for tav unet; b c f h w is for conv3d
    return model_input


def train():
    if config.get('debug_nan', False):
        DebugUnderflowOverflow(model)
        logger.info('NaN debugger registered. Start to detect overflow during training.')
    time_start, last_tic = time.time(), time.time()
    log_buffer = LogBuffer()

    start_step = start_epoch * len(train_dataloader)
    global_step = 0
    total_steps = len(train_dataloader) * config.num_epochs

    load_vae_feat = getattr(train_dataloader.dataset, 'load_vae_feat', False)
    
    t5_path = "output/pretrained_models/t5_ckpts"
    base_ratios = eval(f'ASPECT_RATIO_{config.image_size}_TEST')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t5 = T5Embedder(device=device, local_cache=True, cache_dir=t5_path, torch_dtype=torch.float16, use_offload_folder = "output/pretrained_models/t5_ckpts/use_offload_folder")
    for epoch in range(start_epoch + 1, config.num_epochs + 1):
        data_time_start= time.time()
        data_time_all = 0
        for step, batch in enumerate(train_dataloader):
            # print(f"Epoch: {epoch} Step: {step}")
            if batch[0] == None:
                continue
            bs = batch[0].shape[0]
            # print("shuju dtype: ",batch[0].dtype)
            # batch[0] = batch[0].to(dtype = torch.float16)
            # print("shuju dtype: ",batch[0].dtype)
            data_time_all += time.time() - data_time_start
            if load_vae_feat:
                z = batch[0]
            else:
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=config.mixed_precision == 'fp16'):
                        posterior = low_mem_vae(vae,batch[0])
            clean_images = rearrange(posterior,'b c f h w -> b f c h w')
            # print("clean_images dtype: ",clean_images.dtype)
            y = batch[1]
            # 这儿做caption Embedding
            captions_Embedding = []
            captions_Masks = []
            for caption in y:
                prompts = []
                # prompt_clean, _, hw, ar, custom_hw = prepare_prompt_ar(caption, base_ratios, device=device, show=False)
                prompt_clean, _, hw, ar, custom_hw = prepare_prompt_ar(caption, base_ratios, show=False)
                prompts.append(prompt_clean.strip())
                caption_embs, emb_masks = t5.get_text_embeddings(prompts)
                caption_embs = caption_embs.float()[:, None]
                captions_Embedding.append(caption_embs)
                captions_Masks.append(emb_masks)
            y = torch.cat(captions_Embedding, dim=0) 
            y = y.squeeze(0)
            # y = y.to(dtype = torch.float16)
            # print("caption dtype: ",y.dtype)
            y_mask = torch.cat(captions_Masks, dim=0) 
            # y_mask = y_mask.to(dtype = torch.float16)
            # attention_mask = torch.ones((int(bs*config.need_frames),int(key_tokens)),dtype=torch.float32)
            # Sample a random timestep for each image
            # bs = clean_images.shape[0]
            timesteps = torch.randint(0, config.train_sampling_steps, (bs,), device=clean_images.device).long()
            grad_norm = None
            with accelerator.accumulate(model):
                # Predict the noise residual
                optimizer.zero_grad()
                # print("y_mask",y_mask)
                # print("clean_images",clean_images.shape)
                loss_term = train_diffusion.training_losses(model, clean_images, timesteps, model_kwargs=dict(encoder_hidden_states=y,encoder_attention_mask=y_mask))
                loss = loss_term['loss'].mean()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), config.gradient_clip)
                optimizer.step()
                lr_scheduler.step()
                if accelerator.sync_gradients:
                    ema_update(model_ema, model, config.ema_rate)

            lr = lr_scheduler.get_last_lr()[0]
            logs = {args.loss_report_name: accelerator.gather(loss).mean().item()}
            if grad_norm is not None:
                logs.update(grad_norm=accelerator.gather(grad_norm).mean().item())
            log_buffer.update(logs)
            if (step + 1) % config.log_interval == 0 or (step + 1) == 1:
                t = (time.time() - last_tic) / config.log_interval
                t_d = data_time_all / config.log_interval
                avg_time = (time.time() - time_start) / (global_step + 1)
                eta = str(datetime.timedelta(seconds=int(avg_time * (total_steps - start_step - global_step - 1))))
                eta_epoch = str(datetime.timedelta(seconds=int(avg_time * (len(train_dataloader) - step - 1))))
                log_buffer.average()
                info = f"Step/Epoch [{(epoch-1)*len(train_dataloader)+step+1}/{epoch}][{step + 1}/{len(train_dataloader)}]:total_eta: {eta}, " \
                       f"epoch_eta:{eta_epoch}, time_all:{t:.3f}, time_data:{t_d:.3f}, lr:{lr:.3e}, clean_images:{clean_images.shape}"
                info += ', '.join([f"{k}:{v:.4f}" for k, v in log_buffer.output.items()])
                logger.info(info)
                last_tic = time.time()
                log_buffer.clear()
                data_time_all = 0
            logs.update(lr=lr)
            accelerator.log(logs, step=global_step + start_step)

            global_step += 1
            data_time_start= time.time()
            synchronize()
            if accelerator.is_main_process:
                if ((epoch - 1) * len(train_dataloader) + step + 1) % config.save_model_steps == 0:
                    os.umask(0o000)
                    print("bao cun qian")
                    save_checkpoint(os.path.join(config.work_dir, 'checkpoints'),
                                    epoch=epoch,
                                    step=(epoch - 1) * len(train_dataloader) + step + 1,
                                    model=accelerator.unwrap_model(model),
                                    model_ema=accelerator.unwrap_model(model_ema),
                                    optimizer=optimizer,
                                    lr_scheduler=lr_scheduler
                                    )
                    print("bao cun hou")
            synchronize()

        synchronize()
        if epoch % config.save_model_epochs == 0 or epoch == config.num_epochs:
            if accelerator.is_main_process:
                os.umask(0o000)
                save_checkpoint(os.path.join(config.work_dir, 'checkpoints'),
                                epoch=epoch,
                                step=(epoch - 1) * len(train_dataloader) + step + 1,
                                model=accelerator.unwrap_model(model),
                                model_ema=accelerator.unwrap_model(model_ema),
                                optimizer=optimizer,
                                lr_scheduler=lr_scheduler
                                )
        synchronize()


def parse_args():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("config", type=str, help="config")
    parser.add_argument("--video_length", default=56,help="video_length")
    parser.add_argument("--cloud", action='store_true', default=False, help="cloud or local machine")
    parser.add_argument('--work_dir', default="output/train_InternVId_1_256", help='the dir to save logs and models')
    parser.add_argument('--resume_from', help='the dir to resume the training')
    parser.add_argument('--load_from', help='the dir to load a ckpt for training')
    parser.add_argument('--local-rank', type=int, default=-1)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="text2image-fine-tune",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument("--loss_report_name", type=str, default="loss")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    config = read_config(args.config)
    print("args.work_dir ",args.work_dir)
    print("args.resume_from ",args.resume_from)
    print("args.load_from ",args.load_from)
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        config.work_dir = args.work_dir
    if args.cloud:
        config.data_root = '/data/data'
    if args.resume_from is not None:
        config.load_from = None
        config.resume_from = dict(
            checkpoint=args.resume_from,
            load_ema=False,
            resume_optimizer=True,
            resume_lr_scheduler=True)
    if args.debug:
        config.log_interval = 1
        config.train_batch_size = 8
        config.valid_num = 100

    os.umask(0o000)
    os.makedirs(config.work_dir, exist_ok=True)
    init_handler = InitProcessGroupKwargs()
    init_handler.timeout = datetime.timedelta(seconds=5400)  # change timeout to avoid a strange NCCL bug
    # Initialize accelerator and tensorboard logging
    if config.use_fsdp:
        init_train = 'FSDP'
        from accelerate import FullyShardedDataParallelPlugin
        from torch.distributed.fsdp.fully_sharded_data_parallel import FullStateDictConfig
        set_fsdp_env()
        fsdp_plugin = FullyShardedDataParallelPlugin(state_dict_config=FullStateDictConfig(offload_to_cpu=False, rank0_only=False),)
    else:
        init_train = 'DDP'
        fsdp_plugin = None

    even_batches = True
    if config.multi_scale:
        even_batches=False,
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    print("config.mixed_precision",config.mixed_precision)
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with=args.report_to,
        project_dir=os.path.join(config.work_dir, "logs"),
        fsdp_plugin=fsdp_plugin,
        even_batches=even_batches,
        kwargs_handlers=[init_handler,kwargs]
    )

    logger = get_root_logger(os.path.join(config.work_dir, 'train_log.log'))

    config.seed = init_random_seed(config.get('seed', None))
    set_random_seed(config.seed)

    if accelerator.is_main_process:
        config.dump(os.path.join(config.work_dir, 'config.py'))

    logger.info(f"Config: \n{config.pretty_text}")
    logger.info(f"World_size: {get_world_size()}, seed: {config.seed}")
    logger.info(f"Initializing: {init_train} for training")
    image_size = config.image_size  # @param [256, 512, 1024]
    latent_size = int(image_size) // 8
    pred_sigma = getattr(config, 'pred_sigma', True)
    learn_sigma = getattr(config, 'learn_sigma', True) and pred_sigma
    model_kwargs={"window_block_indexes": config.window_block_indexes, "window_size": config.window_size,
                  "use_rel_pos": config.use_rel_pos, "lewei_scale": config.lewei_scale, 'config':config,
                  'model_max_length': config.model_max_length}

    # build models
    train_diffusion = IDDPM(str(config.train_sampling_steps), learn_sigma=learn_sigma, pred_sigma=pred_sigma, snr=config.snr_loss)
    args.video_length = config.need_frames
    model = get_debug_models(args)
    model = model.train()
    logger.info(f"{model.__class__.__name__} Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    model_ema = deepcopy(model).eval()
    if config.load_from is not None:
        if args.load_from is not None:
            config.load_from = args.load_from
        # model.load_state_dict(torch.load(config.load_from)['ema'])
        # model_ema.load_state_dict(torch.load(config.load_from)['ema'])
        missing, unexpected = load_checkpoint_net2(config.load_from, model, load_ema=config.get('load_ema', False))
        logger.warning(f'Missing keys: {missing}')
        logger.warning(f'Unexpected keys: {unexpected}')

    ema_update(model_ema, model, 0.)
    if not config.data.load_vae_feat:
        vae = AutoencoderKL.from_pretrained(config.vae_pretrained,torch_dtype=torch.float16).cuda()

    # prepare for FSDP clip grad norm calculation
    if accelerator.distributed_type == DistributedType.FSDP:
        for m in accelerator._models:
            m.clip_grad_norm_ = types.MethodType(clip_grad_norm_, m)

    # build dataloader
    set_data_root(config.data_root)
    dataset = build_dataset(config.data, resolution=image_size, aspect_ratio_type=config.aspect_ratio_type, needed_frames = config.need_frames, use_video_num = config.train_video_num)
    print("config.multi_scale",config.multi_scale)
    if config.multi_scale:
        batch_sampler = AspectRatioBatchSampler(sampler=RandomSampler(dataset), dataset=dataset,
                                                batch_size=config.train_batch_size, aspect_ratios=dataset.aspect_ratio, drop_last=True,
                                                ratio_nums=dataset.ratio_nums, config=config, valid_num=config.valid_num)
        # used for balanced sampling
        # batch_sampler = BalancedAspectRatioBatchSampler(sampler=RandomSampler(dataset), dataset=dataset,
        #                                                 batch_size=config.train_batch_size, aspect_ratios=dataset.aspect_ratio,
        #                                                 ratio_nums=dataset.ratio_nums)
        train_dataloader = build_dataloader(dataset, batch_sampler=batch_sampler, num_workers=config.num_workers, batch_size=config.train_batch_size, collate_fn=safe_collate)
    else:
        train_dataloader = build_dataloader(dataset, num_workers=config.num_workers, batch_size=config.train_batch_size, shuffle=True, collate_fn=safe_collate)

    # build optimizer and lr scheduler
    lr_scale_ratio = 1
    if config.get('auto_lr', None):
        lr_scale_ratio = auto_scale_lr(config.train_batch_size * get_world_size() * config.gradient_accumulation_steps,
                                       config.optimizer, **config.auto_lr)
    model = model.cuda()
    optimizer = build_optimizer(model, config.optimizer)
    lr_scale_ratio = 1
    print("lr_scale_ratio: ", lr_scale_ratio)
    config.lr_scale_ratio=lr_scale_ratio
    lr_scheduler = build_lr_scheduler(config, optimizer, train_dataloader, lr_scale_ratio)

    timestamp = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())

    if accelerator.is_main_process:
        tracker_config = dict(vars(config))
        try:
            accelerator.init_trackers(args.tracker_project_name, tracker_config)
        except:
            accelerator.init_trackers(f"tb_{timestamp}")

    start_epoch = 0
    if config.resume_from is not None and config.resume_from['checkpoint'] is not None:
        start_epoch, missing, unexpected = load_checkpoint_net2(**config.resume_from,
                                                           model=model,
                                                           model_ema=model_ema,
                                                           optimizer=optimizer,
                                                           lr_scheduler=lr_scheduler,
                                                           )
        logger.warning(f'Missing keys: {missing}')
        logger.warning(f'Unexpected keys: {unexpected}')
    ##########################################################
    model, model_ema = accelerator.prepare(model, model_ema)
    # model = model.to(dtype = torch.float16)
    #使用deepspeed不能用accelerator实例化两个模型###############
    ##########################################################
    # model = accelerator.prepare(model)
    optimizer, train_dataloader, lr_scheduler = accelerator.prepare(optimizer, train_dataloader, lr_scheduler)
    # import deepspeed
    # model_ema_engine, _, _, _ = deepspeed.initialize(
    #     model=model_ema,
    #     config_params=accelerator.state.deepspeed_plugin.deepspeed_config,
    # )
    # print("accelerator.wait_for_everyone() before")

    # if init_train == 'FSDP':
    #     from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    #     from torch.distributed.fsdp import StateDictType,FullOptimStateDictConfig
    #     FSDP.set_state_dict_type(
    #     model,
    #     StateDictType.FULL_STATE_DICT,
    #     FullStateDictConfig(offload_to_cpu=False, rank0_only=False),
    #     FullOptimStateDictConfig(rank0_only=False))
    train()
