import os
import re
import torch

from diffusion.utils.logger import get_root_logger


def save_checkpoint(work_dir,
                    epoch,
                    model,
                    model_ema=None,
                    optimizer=None,
                    lr_scheduler=None,
                    keep_last=False,
                    step=None,
                    ):
    os.makedirs(work_dir, exist_ok=True)
    state_dict = dict(state_dict=model.state_dict())
    if model_ema is not None:
        state_dict['state_dict_ema'] = model_ema.state_dict()
    if optimizer is not None:
        state_dict['optimizer'] = optimizer.state_dict()
    if lr_scheduler is not None:
        state_dict['scheduler'] = lr_scheduler.state_dict()
    if epoch is not None:
        state_dict['epoch'] = epoch
        file_path = os.path.join(work_dir, f"epoch_{epoch}.pth")
        if step is not None:
            file_path = file_path.split('.pth')[0] + f"_step_{step}.pth"
    logger = get_root_logger()
    torch.save(state_dict, file_path)
    logger.info(f'Saved checkpoint of epoch {epoch} to {file_path.format(epoch)}.')
    if keep_last:
        for i in range(epoch):
            previous_ckgt = file_path.format(i)
            if os.path.exists(previous_ckgt):
                os.remove(previous_ckgt)

def fsdp_save_checkpoint(work_dir,
                    epoch,
                    model,
                    model_ema=None,
                    optimizer=None,
                    lr_scheduler=None,
                    keep_last=False,
                    step=None,
                    ):
    os.makedirs(work_dir, exist_ok=True)
    print("into save_checkpoint")
    state_dict = dict(state_dict=model.state_dict())
    if model_ema is not None:
        state_dict['state_dict_ema'] = model_ema.state_dict()
    if optimizer is not None:
        state_dict['optimizer'] = optimizer.state_dict()
    if lr_scheduler is not None:
        state_dict['scheduler'] = lr_scheduler.state_dict()
    if epoch is not None:
        state_dict['epoch'] = epoch
        file_path = os.path.join(work_dir, f"epoch_{epoch}.pth")
        if step is not None:
            file_path = file_path.split('.pth')[0] + f"_step_{step}.pth"
    logger = get_root_logger()
    torch.save(state_dict, file_path)
    logger.info(f'Saved checkpoint of epoch {epoch} to {file_path.format(epoch)}.')
    if keep_last:
        for i in range(epoch):
            previous_ckgt = file_path.format(i)
            if os.path.exists(previous_ckgt):
                os.remove(previous_ckgt)


def load_checkpoint(checkpoint,
                    model,
                    model_ema=None,
                    optimizer=None,
                    lr_scheduler=None,
                    load_ema=False,
                    resume_optimizer=True,
                    resume_lr_scheduler=True
                    ):
    assert isinstance(checkpoint, str)
    ckpt_file = checkpoint
    checkpoint = torch.load(ckpt_file, map_location="cpu")
    state_dict_keys = ['pos_embed', 'base_model.pos_embed', 'model.pos_embed']
    for key in state_dict_keys:
        if key in checkpoint['state_dict']:
            del checkpoint['state_dict'][key]
            if 'state_dict_ema' in checkpoint and key in checkpoint['state_dict_ema']:
                del checkpoint['state_dict_ema'][key]
            break

    # # 加载原权重才需要调整权重 否则注释掉
    # if 'x_embedder.proj.weight' in checkpoint['state_dict']:
    #     weight = checkpoint['state_dict']['x_embedder.proj.weight']
    #     new_weight = weight.unsqueeze(2)  # 在正确的位置增加所需的维度
    #     checkpoint['state_dict']['x_embedder.proj.weight'] = new_weight

    # 从14帧到16帧输入变化 pos_embed_temporal维度对齐失败 但是pos_embed_temporal梯度未打开 可删除这个key
    del checkpoint['state_dict']["pos_embed_temporal"]

    if load_ema:
        state_dict = checkpoint['state_dict_ema']
    else:
        state_dict = checkpoint.get('state_dict', checkpoint)  # to be compatible with the official checkpoint
    # model.load_state_dict(state_dict)
    missing, unexpect = model.load_state_dict(state_dict, strict=False)
    if model_ema is not None:
        del checkpoint['state_dict_ema']["pos_embed_temporal"]
        model_ema.load_state_dict(checkpoint['state_dict_ema'], strict=False)
    if optimizer is not None and resume_optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if lr_scheduler is not None and resume_lr_scheduler:
        lr_scheduler.load_state_dict(checkpoint['scheduler'])
    logger = get_root_logger()
    if optimizer is not None:
        epoch = checkpoint.get('epoch', re.match(r'.*epoch_(\d*).*.pth', ckpt_file).group()[0])
        logger.info(f'Resume checkpoint of epoch {epoch} from {ckpt_file}. Load ema: {load_ema}, '
                    f'resume optimizer： {resume_optimizer}, resume lr scheduler: {resume_lr_scheduler}.')
        return epoch, missing, unexpect
    logger.info(f'Load checkpoint from {ckpt_file}. Load ema: {load_ema}.')
    return missing, unexpect


def load_checkpoint_net2(checkpoint,
                    model,
                    model_ema=None,
                    optimizer=None,
                    lr_scheduler=None,
                    load_ema=False,
                    resume_optimizer=True,
                    resume_lr_scheduler=True
                    ):
    assert isinstance(checkpoint, str)
    ckpt_file = checkpoint
    checkpoint = torch.load(ckpt_file, map_location="cpu")
    if load_ema:
        state_dict = checkpoint['state_dict_ema']
    else:
        state_dict = checkpoint['state_dict']  # to be compatible with the official checkpoint
    missing, unexpect = model.load_state_dict(state_dict, strict=False)
    if model_ema is not None:
        # 获取 model_ema 的状态字典
        model_ema_state_dict = model_ema.state_dict()
        # 过滤掉在 model_ema 中不存在的键
        pretrained_state_dict = {k: v for k, v in checkpoint['state_dict_ema'].items() if k in model_ema_state_dict}
        # 更新 model_ema 的状态字典
        model_ema_state_dict.update(pretrained_state_dict)
        model_ema.load_state_dict(model_ema_state_dict, strict=False)
    if optimizer is not None and resume_optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if lr_scheduler is not None and resume_lr_scheduler:
        lr_scheduler.load_state_dict(checkpoint['scheduler'])
    logger = get_root_logger()
    if optimizer is not None:
        epoch = checkpoint.get('epoch', re.match(r'.*epoch_(\d*).*.pth', ckpt_file).group()[0])
        logger.info(f'Resume checkpoint of epoch {epoch} from {ckpt_file}. Load ema: {load_ema}, '
                    f'resume optimizer: {resume_optimizer}, resume lr scheduler: {resume_lr_scheduler}.')
        return epoch, missing, unexpect
    logger.info(f'Load checkpoint from {ckpt_file}. Load ema: {load_ema}.')
    return missing, unexpect