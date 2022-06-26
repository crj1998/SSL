""" 
This file build a dataset for semi supervised learning from `data` field of config.yaml 

data:
    root: str
    name: str
    num_classes: int
    batch_size: int
    num_workers: int
    mu: int
    [optional] num_labeled: int
    [optional] eval_step: int

    labeled:
        folder: str
        transforms: List
    unlabeled:
        folder: str
        transforms: List
    test:
        folder: str
        transforms: List
"""

import os
import pickle
from copy import deepcopy
from multiprocessing.pool import ThreadPool
from tqdm import tqdm

import numpy as np
from torchvision import datasets
import torchvision.transforms as T

import dataset.augmentation as augmentation

NUM_THREADS = min(8, max(1, os.cpu_count() - 1))  # number of YOLOv5 multiprocessing threads

def parse_transform(config):
    """
    convert transforms dict list to transforms
    """
    trans = []
    for param in config:
        param = deepcopy(param)
        assert "name" in param, "data config must have key-value pair like `name: RandomHorizontalFlip`"
        name = param.pop("name")
        if name in ["RandAugmentMC", "Cutout", "FreqFliter"]:
            func = getattr(augmentation, name)
        elif hasattr(T, name):
            func = getattr(T, name)
            if name == "RandomApply":
                param.transforms = parse_transform(param.transforms).transforms
        else:
            raise ValueError(f"Unkonown transform name: {name}")
        trans.append(func(**param))
    return T.Compose(trans)



def parse_transforms(config):
    if isinstance(config[0], dict):
        return parse_transform(config)
    else:
        return [parse_transform(i) for i in config]


class ImageFolder(datasets.ImageFolder):
    def __init__(self, root, name="default", return_index=False, cached=False, **kwargs):
        super().__init__(root, **kwargs)
        self.name = name
        self.return_index = return_index
        self.cached = cached        
        if self.cached:
            cache_path = os.path.join(root, f"../{name}.cache")
            if os.path.exists(cache_path):
                with open(cache_path, "rb") as f:
                    self.cache = pickle.load(f)
            else:
                self.cache = []
                results = ThreadPool(NUM_THREADS).imap(self.loader, map(lambda x: x[0], self.samples))
                pbar = tqdm(enumerate(results), total=len(self.samples), disable=False)
                gb = 0
                for i, x in pbar:
                    self.cache.append((x, self.samples[i][1]))
                    gb += np.asarray(x).nbytes
                    pbar.desc = f'Caching images for Dataset {self.name} ({gb / 1E9:.1f} GB)'
                pbar.close()
                with open(cache_path, "wb") as f:
                    pickle.dump(self.cache, f)
    
    def __repr__(self):
        return f"ImageFolder(\n  root={self.root}\n  classes={len(self.classes)}\n  samples={self.__len__()}\n)"
    
    def __getitem__(self, index):
        if self.cached:
            sample, target = self.cache[index]
        else:
            path, target = self.samples[index]        
            sample = self.loader(path)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if isinstance(self.transform, list):
            samples = tuple(tran(sample) for tran in self.transform)
            res =  (*samples, target)
        else:
            sample = sample if self.transform is None else self.transform(sample)
            res = (sample, target)
        
        if self.return_index:
            return (*res, index)
        else:
            return res
        
        

def build(config):
    name = config.name
    root = config.root

    labeled_dataset, unlabeled_dataset, test_dataset = None, None, None
    cached = True if "cifar" in name.lower() or config.labeled.get("cached", False) else False

    if "labeled" in config:
        labeled_transform = parse_transforms(config.labeled.transforms)
        labeled_dataset = ImageFolder(name="Labeled", root=os.path.join(root, config.labeled.folder), transform=labeled_transform, return_index=config.labeled.get("index", False), cached=cached)
    
    if "unlabeled" in config:
        unlabeled_transform = parse_transforms(config.unlabeled.transforms)
        unlabeled_dataset = ImageFolder(name="Unlabeled", root=os.path.join(root, config.unlabeled.folder), transform=unlabeled_transform, cached=cached)
    
    if "test" in config:
        test_transform = parse_transforms(config.test.transforms)
        test_dataset = ImageFolder(name="Test", root=os.path.join(root, config.test.folder), transform=test_transform, cached=cached)

    # print(labeled_transform, unlabeled_transform, test_transform)

    return labeled_dataset, unlabeled_dataset, test_dataset
