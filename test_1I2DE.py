# test one image for different epoch
import argparse
import sys
import random
import os
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
import glob
from models.networks import ResnetGenerator,UnetGenerator
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, '%s/A' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/B' % mode) + '/*.*'))

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))

        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]))
        else:
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
# Networks
def test(name,net_G,cp_dir,epoch_start=5,epoch_end=200,epoch_stride=5,direction="A2B"):
    # 256
    netG_A2B = ResnetGenerator(3, 3,n_blocks=9) if net_G=="resnet" else UnetGenerator(3,3,8)
    if torch.cuda.is_available():
        netG_A2B.cuda()
    for epoch_count in range(epoch_start,epoch_end,epoch_stride):
        netG_A2B.load_state_dict(torch.load("checkpoints/"+cp_dir+"/"+str(epoch_count)+"_net_G_B.pth"))
        Tensor = torch.cuda.FloatTensor
        input_A = Tensor(1, 3, 256, 256)
        input_B = Tensor(1, 3, 256, 256)
        # Dataset loader
        transforms_ = [ transforms.ToTensor(),
                        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
        dataloader = DataLoader(ImageDataset("one_image", transforms_=transforms_, mode='test'),
                                batch_size=1, shuffle=False, num_workers=0)
        for i, batch in enumerate(dataloader):
            real_A = Variable(input_A.copy_(batch['A']))
            # Generate output
            fake_B = 0.5*(netG_A2B(real_A).data + 1.0)
            save_image(fake_B, "one_image/"+name+"/fake_B_epoch_"+str(epoch_count))
        print(str(epoch_count))


if __name__ == "__main__":
    test("shuimo","resnet","shuimo_cyclegan",5,20,5)