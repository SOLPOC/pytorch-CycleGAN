import torch
import torch.nn.functional as F
from torch.autograd import Variable
import math
import os
from skimage import io
import numpy as np
import random
from PIL import Image


def PSNR(img1, img2):
    # pdb.set_trace()
    img1 = np.float64(img1)
    img2 = np.float64(img2)
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
         return 100
    PIXEL_MAX = 255.0
    # PIXEL_MAX = 1.0
    psnr = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    return psnr

def get_PSNR(fake_dir,standard_dir):
    fake_imgs=os.listdir(fake_dir)
    standard_imgs=os.listdir(standard_dir)
    psnr=0.0
    for fake_img in fake_imgs:
        index_s = random.randint(0, len(standard_imgs) - 1)
        fake_img_path=os.path.join(fake_dir,fake_img)
        standard_img_path=os.path.join(standard_dir,standard_imgs[index_s])
        psnr+=PSNR(Image.open(fake_img_path),Image.open(standard_img_path))
    # print(len(fake_imgs))
    return psnr/len(fake_imgs)

def get_mean_PSNR(fake_dir,standard_dir,sum=10):
    s_PSNR=0.0
    for i in range(sum):
        s_PSNR+=get_PSNR(fake_dir,standard_dir)
    return s_PSNR/sum


