"""
Author      : now more
Connect     : lin.honghui@qq.com
LastEditors: Please set LastEditors
Description : 基于Pytorch多线程加载数据集
LastEditTime: 2020-11-27 04:17:38
"""

from ..dataset import build_dataset
from ..transform import build_transforms

import data.dataloader.sampler as Samplers
import data.dataloader.collate_fn as Collate_fn
from torch.utils.data import DataLoader, random_split
from copy import deepcopy


def build_dataloader(cfg_data_pipeline):
    cfg_data_pipeline = deepcopy(cfg_data_pipeline)
    cfg_train_dataset = cfg_data_pipeline.pop("train_dataset")
    cfg_test_dataset = cfg_data_pipeline.pop("test_dataset")
    cfg_transforms = cfg_data_pipeline.pop("transforms")
    cfg_train_loader = cfg_data_pipeline.pop("train_loader")
    cfg_test_loader = cfg_data_pipeline.pop("test_loader")

    transforms = build_transforms(cfg_transforms)
    train_set = build_dataset(cfg_train_dataset, transforms)
    test_set = build_dataset(cfg_test_dataset, transforms)
    if "sampler" in cfg_train_loader:
        cfg_sample = cfg_train_loader.pop("sampler")
        sample_type = cfg_sample.pop("type")
        sampler = getattr(Samplers, sample_type)(train_set.label, **cfg_sample)
        train_loader = DataLoader(train_set, sampler=sampler, **cfg_train_loader)
        test_loader = DataLoader(test_set, sampler=sampler, **cfg_test_loader)

    else:
        if "collate_fn" in cfg_train_loader:
            cfg_collate_fn = cfg_train_loader.pop("collate_fn")
            if hasattr(Collate_fn, cfg_collate_fn):
                collate_fn = getattr(Collate_fn, cfg_collate_fn)
                train_loader = DataLoader(
                    train_set, collate_fn=collate_fn, **cfg_train_loader
                )
                test_loader = DataLoader(
                    test_set, collate_fn=collate_fn, **cfg_test_loader
                )
        else:
            train_loader = DataLoader(train_set, **cfg_train_loader)
            test_loader = DataLoader(test_set, **cfg_test_loader)
    return train_loader, test_loader
