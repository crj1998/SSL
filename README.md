# Introduction
This is a semi-supervised learning toolbox in PyTorch. 

**Supported algorithms**
- &#9745; Supervised baseline
- &#9745; FixMatch (NeurIPS 2020)[1]
- &#9745; CoMatch (ICCV 2021)[2]
- &#9745; SimMatch (CVPR 2022)[3]
- &#9744; CCSSL(CVPR 2022)[4]



## BenchMark

WideResNet-28-2 for CIFAR10, Supervised accuracy of all training samples (50000) is **95.06%**.
WideResNet-28-8 for CIFAR100, Supervised accuracy of all training samples (50000) is **80.70%**.

| Method |       |CIFAR10 |       |       |CIFAR100|      |
|:------:|:-----:|:------:|:-----:|:-----:|:------:|:----:|
|        |  40   |  250   | 4000  |  400  | 2500  | 10000 |
|Supervised|     | 43.63 |   |    |   |  |
|FixMatch|     | 94.53 |   |    | 72.43 |  |
|CoMatch |     | 94.08 |   |    |   |  |
|SimMatch|     | 94.60 |   |    |   |  |


CIFAR10-100: 94.14

### Hyper-parameter for CIFAR10

tau=0.95, mu=7, lambda_mu=1.0, B=64, 

optimizer: lr=0.03, beta=0.9, weight decay=5e^{-4}, Nesterov used.

scheduler: cosine learning rate decay, with total steps: 2^{17} = 1024*128.


## Usage
### Install and Setup
Clone this repo to your machine and install dependencies:  
We use torch==1.8.0 and torchvision==0.9.0 for CUDA 11.0

```
conda create -n ssl python=3.8
conda activate ssl
pip install -r requirements.txt
```
### Training
1. **prepare datasets**  
Organize your datasets as the following ImageFolder format:

```
data
├── CIFAR10
│   ├── labeled
│   │   ├── 0
│   │   │   ├── xxx.png
│   │   │   ├── xxx.png
│   │   │   └── xxx.png
│   │   ├── x
│   │   │   ├── xxx.png
│   │   │   ├── xxx.png
│   │   │   └── xxx.png
│   │   └── 9
│   │       ├── xxx.png
│   │       └── xxx.png
│   ├── unlabeled
│   │   ├── 0
│   │   │   ├── xxx.png
│   │   │   ├── xxx.png
│   │   │   └── xxx.png
│   │   ├── x
│   │   │   ├── xxx.png
│   │   │   ├── xxx.png
│   │   │   └── xxx.png
│   │   └── 9
│   │       ├── xxx.png
│   │       └── xxx.png
│   ├── test
│   │   ├── 0
│   │   │   ├── xxx.png
│   │   │   ├── xxx.png
│   │   │   └── xxx.png
│   │   ├── x
│   │   │   ├── xxx.png
│   │   │   ├── xxx.png
│   │   │   └── xxx.png
│   │   └── 9
│   │       ├── xxx.png
│   │       └── xxx.png
│   └── cifar-100-python # cifar100
└── customdata
    ├── labeled
    │   ├── 0
    │   │   ├── xxx.png
    │   │   ├── xxx.png
    │   │   └── xxx.png
    │   ├── x
    │   │   ├── xxx.png
    │   │   ├── xxx.png
    │   │   └── xxx.png
    │   └── 9
    │       ├── xxx.png
    │       └── xxx.png
    ├── unlabeled
    │   └── 0
    │       ├── xxx.png
    │       ├── xxx.png
    │       ├── xxx.png
    │       ├── xxx.png
    │       └── xxx.png
    └── test
        ├── 0
        │   ├── xxx.png
        │   ├── xxx.png
        │   └── xxx.png
        ├── x
        │   ├── xxx.png
        │   ├── xxx.png
        │   └── xxx.png
        └── 9
            ├── xxx.png
            └── xxx.png
```

```
# convert pytorch CIFAR10/CIFAR100 format to ImageFolder
python tools/make_cifar.py --dataset CIFAR10 --src_path path/to --tar_path path/to --labeled_per_class 100
```
2. **make config file**
Modify the config file in `SSL/configs` folder. Like set dataset, backbone, algorithm, hyper-parameters.

3. **start train**
Now you can run the experiments for different SSL althorithms by modifying configs as you need.  

```
## Single-GPU
# train the model by 40 labeled data of CIFAR-10 dataset by FixMatch:
python main.py --cfg configs/dev.yaml --out path/to --seed 5 --gpu 0

## Multi-GPU
# train the model by CIFAR100 dataset by FixMatch+CCSSL with 4GPUs:
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 main.py --cfg configs/dev.yaml --out results/dev --use_BN True --seed 5
```


## Customization
The Framework use `config.yaml` to prepare `model(include EMA)`, `dataloader`, `optimizer`, `scheduler`. The `Trainer` is used to calculate ssl loss. 



## Reference

1. Fixmatch: Simplifying semi-supervised learning with consistency and confidence. NeurIPS 2020.
2. Comatch: Semi-supervised learning with contrastive graph regularization. ICCV 2021.
3. SimMatch: Semi-supervised Learning with Similarity Matching. CVPR 2022.
4. Class-Aware Contrastive Semi-Supervised Learning. CVPR 2022.
