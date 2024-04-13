import torchvision.models as models
import torch.nn as nn
from torchinfo import summary
from models.networks import UnetGenerator
# resnet18 = models.resnet18() # 实例化模型
# print(resnet18)
unet=UnetGenerator( input_nc=3, output_nc=3, num_downs=8, ngf=64, norm_layer=nn.InstanceNorm2d, use_dropout=False)
print(unet)
summary(unet, (1, 3, 224, 224)) # 1：batch_size 3:图片的通道数 224: 图片的高宽