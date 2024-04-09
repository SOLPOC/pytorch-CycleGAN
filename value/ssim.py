import torch
import torch.nn.functional as F
from torch.autograd import Variable
import math
import os
from skimage import io
import numpy as np
import random
from PIL import Image


def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def SSIM(img1,img2):
	# pdb.set_trace()
	img1 = torch.from_numpy(np.rollaxis(img1, 2)).float().unsqueeze(0)/255.0
	img2 = torch.from_numpy(np.rollaxis(img2, 2)).float().unsqueeze(0)/255.0
	img1 = Variable( img1,  requires_grad=False)    # torch.Size([256, 256, 3])
	img2 = Variable( img2, requires_grad = False)
	# ssim_value = pytorch_ssim.ssim(img1, img2).item()
	ssim_value = float(ssim(img1, img2))
	return ssim_value

def get_SSIM(fake_dir,standard_dir):
    fake_imgs=os.listdir(fake_dir)
    standard_imgs=os.listdir(standard_dir)
    ssim=0.0
    for fake_img in fake_imgs:
        index_s = random.randint(0, len(standard_imgs) - 1)
        fake_img_path=os.path.join(fake_dir,fake_img)
        standard_img_path=os.path.join(standard_dir,standard_imgs[index_s])
        ssim+=SSIM(io.imread(fake_img_path),io.imread(standard_img_path))
    return ssim/len(fake_imgs)

def get_mean_SSIM(fake_dir,standard_dir,sum=10):
    s_SSIM=0.0
    for i in range(sum):
        s_SSIM+=get_SSIM(fake_dir,standard_dir)
    return s_SSIM/sum


# from skimage.metrics import structural_similarity as SSIM
# from skimage.metrics import peak_signal_noise_ratio as PSNR
# from skimage import io

# ssim = SSIM(img1, img2, multichannel=True, channel_axis=2)
