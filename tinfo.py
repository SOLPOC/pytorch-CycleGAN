import torchvision.models as models
import torch.nn as nn
import torch
from torchinfo import summary
from models.networks import *
from models.networks import SAUnetGenerator as Generator
# resnet18 = models.resnet18() # 实例化模型
# print(resnet18)
# unet=UnetGenerator( input_nc=3, output_nc=3, num_downs=8, ngf=64, norm_layer=nn.InstanceNorm2d, use_dropout=False)
# print(unet)

def print_model():
    model=torch.load("checkpoints/painting_saunet_old/latest_net_G.pth")
    print(model)
def load_model(model_path):
    model = torch.load(model_path, map_location=lambda storage, loc: storage)
    print(model)
    return model

print_model()
# model = load_model("checkpoints/painting_saunet_old/test.pth")
# summary(unet, (1, 3, 224, 224)) # 1：batch_size 3:图片的通道数 224: 图片的高宽
# if __name__ == '__main__':
#     print_model()