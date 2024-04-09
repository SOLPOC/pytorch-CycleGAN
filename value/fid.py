"""
python -m pytorch_fid path/to/dataset1 path/to/dataset2
"""
"""
conda activate torch
cd /d D:\O\Python\pytorch-CycleGAN-and-pix2pix-master
python -m pytorch_fid datasets/maps/testB results/sat2map_pretrained/test_latest/images/B_fake  --device cuda:0
python -m pytorch_fid results/sat2map_pretrained/test_latest/images/B_fake  results/sat2map_pretrained/test_latest/images/B_fake --device cuda:0
python -m pytorch_fid results/sat2map_pretrained/test_latest/images/B_real results/sat2map_pretrained/test_latest/images/B_fake  --device cuda:0
python -m pytorch_fid datasets/maps/testB datasets/maps/trainB  --device cuda:0
python -m pytorch_fid datasets/maps/testB datasets/horse2zebra/trainB  --device cuda:0
python -m pytorch_fid results/sat2map_pretrained/test_latest/images/B_fake results/sat2map_pretrained/test_latest/images/A_real  --device cuda:0
"""
import torch
import torchvision
import torchvision.transforms as transforms
from pytorch_fid import fid_score

# 准备真实数据分布和生成模型的图像数据
real_images_folder = '../datasets/maps/testB'
generated_images_folder = '../results/sat2map_pretrained/test_latest/images/B_fake'

# 加载预训练的Inception-v3模型
inception_model = torchvision.models.inception_v3(pretrained=True)

# 定义图像变换
transform = transforms.Compose([
    transforms.Resize(286),
    transforms.RandomCrop(256),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 计算FID距离值
fid_value = fid_score.calculate_fid_given_paths([real_images_folder, generated_images_folder],batch_size=1,device="cuda",dims=2
                                                )
print('FID value:', fid_value)

