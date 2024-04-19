import torch
from torchvision import datasets, transforms
from PIL import Image
import os
from models.networks import *
from models.networks import SAUnetGenerator as Generator

# 加载模型
import torch
from torchvision import datasets, transforms
from PIL import Image
import os

# 检查是否有可用的GPU
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# # 加载模型并移动到GPU
# model = torch.load("./checkpoints/painting_saunet_old/latest_net_G.pth")
# model = model.to(device)

# -*- coding: utf-8 -*-
"""cycleunetattngan.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1gfzDj-R3XYXSWLR3sf0mI_gybx_5OEGi
"""

import glob
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as L
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from pytorch_lightning.trainer.supporters import CombinedLoader
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
from torchvision.utils import make_grid, save_image


import torch
from torchvision import datasets, transforms
from PIL import Image
import os

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载模型并移动到GPU
model = torch.load("checkpoints/painting_saunet_old/latest_net_G.pth")
model = model.to(device)

# 定义转换操作
transform = transforms.Compose([
    transforms.Resize((286,286)),
    transforms .RandomCrop((256,256)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2,
                  saturation=0.2, hue=0.1),
    transforms.ToTensor(),
])

# 定义输入图片的文件夹路径
input_folder = "datasets/vangogh2photo/testA"

# 定义输出图片的文件夹路径
output_folder = "res"
os.makedirs(output_folder, exist_ok=True)

# 遍历输入文件夹中的每一张图片
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # 加载并转换图片
        img = Image.open(os.path.join(input_folder, filename))
        img = transform(img).unsqueeze(0)
        img = img.to(device)  # 将输入移动到GPU

        # 使用模型生成输出
        with torch.no_grad():
            output = model(img)

        # 将输出转换回PIL图片并保存
        output_img = transforms.ToPILImage()(output.squeeze().cpu())  # 将输出移动回CPU
        output_img.save(os.path.join(output_folder, filename))

print("所有图片处理完成！")