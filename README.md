
# Introduction
This repository contains a minimal implementation of Neural Radiance Fields (NeRF) using PyTorch. NeRF is a powerful method for 3D scene representation and rendering, enabling the creation of high-quality synthetic views of complex scenes from a set of 2D images.
This program will output images and also the mesh.

![description image](description/nerf.png)


# How to run:
Was run on:
`torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116`

But also works with `torch==2.2.0+cu212`

```
Put your datasets into the data folder/

python3 train.py --save-root checkpoints --data-root data/folder_name
python3 train.py --save-root checkpoints --data-root data/folder_name


python3 test.py --pretrained-root checkpoints/mp-2024/{checkpoint} --model-name model-{epoch}.pth --data-root data/public
python3 test.py --pretrained-root checkpoints/mp-2024/{checkpoint} --model-name model-{epoch}.pth --data-root data/public
```
You can tune hyperparameters in `config.yaml`
