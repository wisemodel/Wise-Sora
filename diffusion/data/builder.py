import os
import time

from mmcv import Registry, build_from_cfg
from torch.utils.data import DataLoader

from diffusion.data.transforms import get_transform
from diffusion.utils.logger import get_root_logger

DATASETS = Registry('datasets')

DATA_ROOT = '/cache/data'


def set_data_root(data_root):
    global DATA_ROOT
    DATA_ROOT = data_root


def get_data_path(data_dir):
    if os.path.isabs(data_dir):
        return data_dir
    global DATA_ROOT
    return os.path.join(DATA_ROOT, data_dir)


def build_dataset(cfg, resolution=224, **kwargs):
    logger = get_root_logger()

    dataset_type = cfg.get('type')
    logger.info(f"Constructing dataset {dataset_type}...")
    t = time.time()
    transform = cfg.pop('transform', 'default_train')
    transform = get_transform(transform, resolution)
    dataset = build_from_cfg(cfg, DATASETS, default_args=dict(transform=transform, resolution=resolution, **kwargs))
    logger.info(f"Dataset {dataset_type} constructed. time: {(time.time() - t):.2f} s, length (use/ori): {len(dataset)}/{dataset.ori_imgs_nums}")
    return dataset


def build_dataloader(dataset, batch_size=256, num_workers=4, shuffle=True, **kwargs):
    # 提取 collate_fn 并确保它不通过 kwargs 传递到 DataLoader
    collate_fn = kwargs.pop('collate_fn', None)

    # 根据是否指定 batch_sampler 来调用 DataLoader
    if 'batch_sampler' in kwargs:
        batch_sampler = kwargs.pop('batch_sampler')
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn  # 明确地传递 collate_fn
        )
    else:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn,  # 明确地传递 collate_fn
            **kwargs  # 传递其他可能的参数
        )

