'''
Author      : now more
Connect     : lin.honghui@qq.com
LastEditors: Please set LastEditors
Description : 基于Pytorch多线程加载数据集
LastEditTime: 2020-11-27 04:17:38
'''

from ..dataset import build_dataset
from ..transform import build_transforms

import data.dataloader.sampler as Samplers
import data.dataloader.collate_fn as Collate_fn
from torch.utils.data import DataLoader,random_split
from copy import deepcopy

def build_dataloader(cfg_data_pipeline):
    cfg_data_pipeline = deepcopy(cfg_data_pipeline)
    cfg_dataset = cfg_data_pipeline.pop('dataset')
    cfg_transforms = cfg_data_pipeline.pop('transforms')
    cfg_train_loader = cfg_data_pipeline.pop('train_loader')
    cfg_val_loader = cfg_data_pipeline.pop('val_loader')

    transforms = build_transforms(cfg_transforms)
    dataset = build_dataset(cfg_dataset,transforms)
    trainlen = int(0.9*len(dataset))
    lengths = [trainlen, len(dataset)-trainlen]
    train_set, val_set = random_split(dataset, lengths)
    if 'sampler' in cfg_train_loader:
        cfg_sample = cfg_train_loader.pop('sampler')
        sample_type = cfg_sample.pop('type')
        sampler = getattr(Samplers,sample_type)(dataset.label,**cfg_sample)
        train_loader = DataLoader(train_set,sampler=sampler,**cfg_train_loader)
        val_loader = DataLoader(val_set,sampler=sampler,**cfg_train_loader)
        
    else:
        if "collate_fn" in cfg_train_loader:
            cfg_collate_fn = cfg_train_loader.pop("collate_fn")
            if hasattr(Collate_fn,cfg_collate_fn):
                collate_fn = getattr(Collate_fn,cfg_collate_fn)
                train_loader  = DataLoader(train_set,collate_fn=collate_fn,**cfg_train_loader)
                val_loader  = DataLoader(val_set,collate_fn=collate_fn,**cfg_train_loader)
        else:
            train_loader  = DataLoader(train_set,**cfg_train_loader)
            val_loader  = DataLoader(val_set,**cfg_train_loader)
    print(len(train_set))
    print(len(val_set))
    return train_loader, val_loader