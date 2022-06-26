"""
$ python tools/make_cifar.py --dataset CIFAR10 --src_path ../../data --tar_path ../../data/SSLdata/CIFAR10_100 --labeled_per_class 100
$ python3.8 tools/make_cifar.py --dataset CIFAR10 --src_path ../../data --tar_path ../../data/SSLdata/CIFAR10_250 --labeled_per_class 25
$ python3.8 tools/make_cifar.py --dataset CIFAR10 --src_path ../../data --tar_path ../../data/SSLdata/CIFAR10_5000 --labeled_per_class 5000
python3.8 tools/make_cifar.py --dataset CIFAR10 --src_path ../../data --tar_path ../../data/SSLdata/CIFAR10_40 --labeled_per_class 4

python3.8 tools/make_cifar.py --dataset CIFAR10 --src_path ../data --tar_path ../../data/SSLdata/CIFAR10 --labeled_per_class 5000
python3.8 tools/make_cifar.py --dataset CIFAR100 --src_path ../data --tar_path ../../data/SSLdata/CIFAR100 --labeled_per_class 500
python3.8 tools/make_cifar.py --dataset CIFAR100 --src_path ../data --tar_path ../../data/SSLdata/CIFAR100_250 --labeled_per_class 25
python3.8 tools/make_cifar.py --dataset CIFAR10 --src_path ../data --tar_path ../../data/SSLdata/CIFAR10_100 --labeled_per_class 10 --expands 6

"""


import os
import shutil
import argparse
from PIL import Image
from collections import defaultdict
from tqdm import tqdm

import numpy as np

from torchvision.datasets import CIFAR10, CIFAR100

parser = argparse.ArgumentParser(description="Convert CIFAR to ImageFolder")
parser.add_argument('--dataset', type=str, required=True, choices=["CIFAR10", "CIFAR100"], help='folder for cifar raw')
parser.add_argument('--src_path', type=str, required=True, help='folder for cifar raw')
parser.add_argument('--tar_path', type=str, required=True, help='folder for cifar folder')
parser.add_argument('--labeled_per_class', default=100, type=int, help="random seed")
parser.add_argument('--expands', default=1, type=int, help="expand labeled data")
parser.add_argument('--included', default=False, action="store_true", help="labeled samples are included in unlabeled samples")
parser.add_argument('--download', default=False, action="store_true", help="download if not exists")

args = parser.parse_args()

num_train_samples = 50000
num_class = int(args.dataset[5:])


if os.path.exists(args.tar_path):
    print(f"Target folder {args.tar_path} existed! Start clear folder!")
    shutil.rmtree(args.tar_path)
    print(f"Folder cleared!")

dataset = eval(args.dataset)(args.src_path, train=True, download=args.download)

for split in ["labeled", "unlabeled", "test"]:
    for label in range(num_class):
        folder = os.path.join(args.tar_path, split, str(label))
        os.makedirs(folder, exist_ok=True)

cnt = defaultdict(int)
shuffle = np.random.permutation(np.arange(num_train_samples))
dataset.data = dataset.data[shuffle]
dataset.targets = np.array(dataset.targets)[shuffle]
for idx, image, label in tqdm(zip(shuffle, dataset.data, dataset.targets), desc="Train"):
    if cnt[label] < args.labeled_per_class:
        split = "labeled"
    else:
        split = "unlabeled"
    im = Image.fromarray(image)
    folder = os.path.join(args.tar_path, split, str(label))
    im.save(os.path.join(folder, f"{idx}.png"))
    if args.included:
        im.save(os.path.join(args.tar_path, "unlabeled", str(label), f"{idx}.png"))
    if split == "labeled":
        for i in range(1, args.expands):
            im.save(os.path.join(folder, f"{idx}_{i}.png"))
    cnt[label] += 1


dataset = eval(args.dataset)(args.src_path, train=False, download=args.download)
split = "test"
i = 0
for image, label in tqdm(zip(dataset.data, dataset.targets), desc="Test"):
    im = Image.fromarray(image)
    folder = os.path.join(args.tar_path, split, str(label))
    im.save(os.path.join(folder, f"{i}.png"))
    i += 1

