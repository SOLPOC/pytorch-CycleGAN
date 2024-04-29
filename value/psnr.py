import torch
import torch.nn.functional as F
from torch.autograd import Variable
import math
import os
from skimage import io
import numpy as np
import random
from PIL import Image
import os
import cv2
from skimage.metrics import structural_similarity as ssim


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

def get_psnr(fake_folder,real_folder):
    # 获取文件夹中的图片文件列表
    fake_files = os.listdir(fake_folder)
    real_files = os.listdir(real_folder)
    t_score = 0
    # 确保文件列表长度一致
    assert len(fake_files) == len(real_files)

    for fake_file, real_file in zip(fake_files, real_files):
        # 读取 fake 图片和 real 图片
        fake_path = os.path.join(fake_folder, fake_file)
        real_path = os.path.join(real_folder, real_file)

        fake_img = cv2.imread(fake_path)
        real_img = cv2.imread(real_path)

        # 应用函数 calculate_ssim 计算 SSIM
        psnr = PSNR(fake_img, real_img)
        t_score += psnr
        # 输出结果（这里可以根据需要保存到文件或进行其他处理）
        print(f"For pair: {fake_file} and {real_file}, SSIM score is: {psnr}")
    return t_score / len(fake_files)


if __name__ == '__main__':
    psnr = get_psnr("../results/map_saunet_a2b/test_latest/images/A_real",
                    "../results/map_saunet_a2b/test_latest/images/B_fake")
    print(psnr)


