import glob
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as L
import torch
import torchvision.transforms as T
from pytorch_lightning.trainer.supporters import CombinedLoader
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
from torchvision.utils import make_grid, save_image

L.seed_everything(0, workers=True)
print(L.__name__, L.__version__)

#data preprocessing
def show_img(img_tensor, nrow, title=""):
    img_tensor = img_tensor.detach().cpu()*0.5 + 0.5
    img_grid = make_grid(img_tensor, nrow=nrow).permute(1, 2, 0)
    plt.figure(figsize=(18, 8))
    plt.imshow(img_grid)
    plt.axis("off")
    plt.title(title)
    plt.show()

class CustomTransform(object):
    def __init__(self, load_dim=286, target_dim=256):
        self.transform_train = T.Compose([
            T.Resize((load_dim, load_dim)),
            T.RandomCrop((target_dim, target_dim)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2,
                          saturation=0.2, hue=0.1),
        ])

        # ensure images outside of training dataset are also of the same size
        self.transform = T.Resize((target_dim, target_dim))

    def __call__(self, img, stage="fit"):
        if stage == "fit":
            img = self.transform_train(img)
        else:
            img = self.transform(img)
        return img*2 - 1

class CustomDataset(Dataset):
    def __init__(self, filenames, transform, stage):
        self.filenames = filenames
        self.transform = transform
        self.stage = stage

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = self.filenames[idx]
        img = read_image(img_name) / 255.0
        return self.transform(img, stage=self.stage)
# A_DIR = "datasets/vtest/trainA/*.jpg"
# B_DIR = "datasets/vtest/trainB/*.jpg"
A_DIR = "datasets/vangogh2photo/testB/*.jpg"
B_DIR = "datasets/vangogh2photo/testA/*.jpg"
DEBUG = not torch.cuda.is_available()
BATCH_SIZE = [1, 4] # [batch size for Monet paintings, batch size for bs]
LOADER_CONFIG = {
    # "num_workers": os.cpu_count(),
    "num_workers": 0,
    "pin_memory": torch.cuda.is_available(),
}

class CustomDataModule(L.LightningDataModule):
    def __init__(
        self,
        debug=DEBUG,
        a_dir=A_DIR,
        b_dir=B_DIR,
        batch_size=BATCH_SIZE,
        loader_config=LOADER_CONFIG,
        transform=CustomTransform(),
        mode="max_size_cycle",
    ):
        super().__init__()
        if isinstance(batch_size, list):
            self.batch_size = batch_size
        else:
            self.batch_size = [batch_size] * 2
        if debug:
            idx = max(self.batch_size) * 2
        else:
            idx = None
        self.a_filenames = sorted(glob.glob(a_dir))[:idx]
        self.b_filenames = sorted(glob.glob(b_dir))[:idx]
        self.loader_config = loader_config
        self.transform = transform
        self.mode = mode

    def setup(self, stage):
        if stage == "fit":
            self.train_a = CustomDataset(self.a_filenames, self.transform, stage)
            self.train_b = CustomDataset(self.b_filenames, self.transform, stage)

        elif stage == "predict":
            self.predict = CustomDataset(self.b_filenames, self.transform, stage)

    def train_dataloader(self):
        loader_a = DataLoader(
            self.train_a,
            shuffle=True,
            drop_last=True,
            batch_size=self.batch_size[0],
            **self.loader_config,
        )
        loader_b = DataLoader(
            self.train_b,
            shuffle=True,
            drop_last=True,
            batch_size=self.batch_size[1],
            **self.loader_config,
        )
        loaders = {"a": loader_a, "b": loader_b}
        return CombinedLoader(loaders, mode=self.mode)

    def predict_dataloader(self):
        return DataLoader(
            self.predict,
            shuffle=False,
            drop_last=False,
            batch_size=self.batch_size[1],
            **self.loader_config,
        )

SAMPLE_SIZE = 5
dm_sample = CustomDataModule(batch_size=SAMPLE_SIZE)

dm_sample.setup("fit")
train_loader = dm_sample.train_dataloader()
a_samples, b_samples = next(iter(train_loader)).values()

dm_sample.setup("predict")
predict_loader = dm_sample.predict_dataloader()
b__samples = next(iter(predict_loader)) # used to track performance of model during training later

# show_img(a_samples, nrow=SAMPLE_SIZE, title="Augmented digital arts")
# show_img(b_samples, nrow=SAMPLE_SIZE, title="Augmented bs")

"""***Lets Define Self Attention Modules***"""

import torch
import torch.nn.functional as F
from torch import nn
class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        # for _ in range(self.power_iterations):
        #     v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
        #     u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data.float()), u.data.float()))
            u.data = l2normalize(torch.mv(w.view(height,-1).data.float(), v.data.float()))
        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)

from torch.nn import Parameter
def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)

        out = self.gamma*out + x
        return out

"""***Lets Define Our Cycle GAN***"""

class Downsampling(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=4,
        stride=2,
        padding=1,
        norm=True,
        attention = False
    ):
        super().__init__()
        self.block = nn.Sequential()
        if attention:
            self.block.append(Self_Attn(in_channels,'relu'))
        self.block.append(nn.Conv2d(in_channels,out_channels,kernel_size = kernel_size,stride=stride,padding = padding,bias= not norm))
        if norm:
            # self.block.append(nn.InstanceNorm2d(out_channels,affine = True))
            self.block[-1]=SpectralNorm(self.block[-1])
            # self.block[-1]= nn.utils.spectral_norm(self.block[-1])
        self.block.append(nn.LeakyReLU(0.3))


    def forward(self,x):
        return self.block(x)

class Upsampling(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=4,
        stride=2,
        padding=1,
        output_padding=0,
        dropout=False,
        attention = False
    ):
        super().__init__()
        self.block = nn.Sequential(
            # nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
            #                    padding=padding, output_padding=output_padding, bias=False),
            # nn.InstanceNorm2d(out_channels, affine=True),
            SpectralNorm( nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=False))
            # nn.utils.spectral_norm(SpectralNorm(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
            #                             padding=padding, output_padding=output_padding, bias=False)))
        )
        if dropout:
            self.block.append(nn.Dropout(0.5))
        self.block.append(nn.ReLU())
        if attention:
            self.block.append(Self_Attn(out_channels,'relu'))




    def forward(self, x):
        return self.block(x)

class Generator(nn.Module):
    def __init__(self, in_channels, out_channels, hid_channels):
        super().__init__()
        self.downsampling_path = nn.Sequential(
            Downsampling(in_channels, hid_channels, norm=False), # 64x128x128
            Downsampling(hid_channels, hid_channels*2), # 128x64x64
            Downsampling(hid_channels*2, hid_channels*4), # 256x32x32
            Downsampling(hid_channels*4, hid_channels*8,attention=True), # 512x16x16
            Downsampling(hid_channels*8, hid_channels*8), # 512x8x8
            Downsampling(hid_channels*8, hid_channels*8), # 512x4x4
            Downsampling(hid_channels*8, hid_channels*8), # 512x2x2
            Downsampling(hid_channels*8, hid_channels*8, norm=False), # 512x1x1, instance norm does not work on 1x1
        )
        self.upsampling_path = nn.Sequential(
            Upsampling(hid_channels*8, hid_channels*8, dropout=True), # (512+512)x2x2
            Upsampling(hid_channels*16, hid_channels*8, dropout=True), # (512+512)x4x4
            Upsampling(hid_channels*16, hid_channels*8, dropout=True), # (512+512)x8x8
            Upsampling(hid_channels*16, hid_channels*8), # (512+512)x16x16
            Upsampling(hid_channels*16, hid_channels*4,attention=True), # (256+256)x32x32
            Upsampling(hid_channels*8, hid_channels*2), # (128+128)x64x64
            Upsampling(hid_channels*4, hid_channels), # (64+64)x128x128
        )
        self.feature_block = nn.Sequential(
            nn.ConvTranspose2d(hid_channels*2, out_channels,
                               kernel_size=4, stride=2, padding=1), # 3x256x256
            nn.Tanh(),
        )

    def forward(self, x):
        skips = []
        for down in self.downsampling_path:
            x = down(x)
            skips.append(x)
        skips = reversed(skips[:-1])

        for up, skip in zip(self.upsampling_path, skips):
            x = up(x)
            x = torch.cat([x, skip], dim=1)
        return self.feature_block(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels, hid_channels):
        super().__init__()
        self.block = nn.Sequential(
            Downsampling(in_channels, hid_channels, norm=False), # 64x128x128
            Downsampling(hid_channels, hid_channels*2), # 128x64x64
            Downsampling(hid_channels*2, hid_channels*4), # 256x32x32
            Downsampling(hid_channels*4, hid_channels*8, stride=1), # 512x31x31
            nn.Conv2d(hid_channels*8, 1, kernel_size=4, padding=1), # 1x30x30
        )

    def forward(self, x):
        return self.block(x)

IN_CHANNELS = 3
OUT_CHANNELS = 3
HID_CHANNELS = 64
LR = 2e-4
BETAS = (0.5, 0.999)
LAMBDA_ID = 2
LAMBDA_CYCLE = 5
NUM_EPOCHS = 30 if not DEBUG else 2
DECAY_EPOCHS = 15 if not DEBUG else 1
DISPLAY_EPOCHS = 5

class CycleGAN(L.LightningModule):
    def __init__(
        self,
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        hid_channels=HID_CHANNELS,
        lr=LR,
        betas=BETAS,
        lambda_id=LAMBDA_ID,
        lambda_cycle=LAMBDA_CYCLE,
        num_epochs=NUM_EPOCHS,
        decay_epochs=DECAY_EPOCHS,
        display_epochs=DISPLAY_EPOCHS,
        b_samples=b__samples,
    ):
        super().__init__()
        self.lr = lr
        self.betas = betas
        self.lambda_id = lambda_id
        self.lambda_cycle = lambda_cycle
        self.num_epochs = num_epochs
        self.decay_epochs = decay_epochs
        self.display_epochs = display_epochs
        self.b_samples = b_samples.to("cuda" if torch.cuda.is_available() else "cpu")

        # record learning rates and losses
        self.lr_history = [self.lr]
        self.loss_names = ["gen_loss_BA", "gen_loss_AB", "disc_loss_A", "disc_loss_B"]
        self.loss_history = {loss: [] for loss in self.loss_names}

        # initialize generators and discriminators
        self.gen_BA = Generator(in_channels, out_channels, hid_channels)
        self.gen_AB = Generator(in_channels, out_channels, hid_channels)
        self.disc_A = Discriminator(in_channels, hid_channels)
        self.disc_B = Discriminator(in_channels, hid_channels)
        self.init_weights()

    def forward(self, z):
        return self.gen_BA(z)


    def init_weights(self):
        def init_fn(m):
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                if hasattr(m, 'weight_bar'):
                    # 如果是SpectralNorm层，初始化其权重参数
                    nn.init.normal_(m.weight_bar, 0.0, 0.02)
                else:
                    nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.InstanceNorm2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                nn.init.constant_(m.bias, 0.0)

        self.gen_BA = self.gen_BA.apply(init_fn)
        self.gen_AB = self.gen_AB.apply(init_fn)
        self.disc_A = self.disc_A.apply(init_fn)
        self.disc_B = self.disc_B.apply(init_fn)

    def adv_criterion(self, y_hat, y):
        return F.mse_loss(y_hat, y)

    def recon_criterion(self, y_hat, y):
        return F.l1_loss(y_hat, y)

    def adv_loss(self, fake_Y, disc_Y):
        fake_Y_hat = disc_Y(fake_Y)
        valid = torch.ones_like(fake_Y_hat)
        adv_loss_XY = self.adv_criterion(fake_Y_hat, valid)
        return adv_loss_XY

    def id_loss(self, real_Y, gen_XY):
        id_Y = gen_XY(real_Y)
        id_loss_Y = self.recon_criterion(id_Y, real_Y)
        return self.lambda_id * id_loss_Y

    def cycle_loss(self, real_Y, fake_X, gen_XY):
        cycle_Y = gen_XY(fake_X)
        cycle_loss_Y = self.recon_criterion(cycle_Y, real_Y)
        return self.lambda_cycle * cycle_loss_Y

    def gen_loss(self, real_X, real_Y, gen_XY, gen_YX, disc_Y):
        fake_Y = gen_XY(real_X)
        fake_X = gen_YX(real_Y)

        adv_loss_XY = self.adv_loss(fake_Y, disc_Y)
        id_loss_Y = self.id_loss(real_Y, gen_XY)
        cycle_loss_Y = self.cycle_loss(real_Y, fake_X, gen_XY)
        cycle_loss_X = self.cycle_loss(real_X, fake_Y, gen_YX)
        total_cycle_loss = cycle_loss_X + cycle_loss_Y

        gen_loss_XY = adv_loss_XY + id_loss_Y + total_cycle_loss
        return gen_loss_XY

    def disc_loss(self, real_Y, fake_Y, disc_Y):
        real_Y_hat = disc_Y(real_Y)
        valid = torch.ones_like(real_Y_hat)
        real_loss_Y = self.adv_criterion(real_Y_hat, valid)

        fake_Y_hat = disc_Y(fake_Y.detach())
        fake = torch.zeros_like(fake_Y_hat)
        fake_loss_Y = self.adv_criterion(fake_Y_hat, fake)

        disc_loss_Y = (fake_loss_Y+real_loss_Y) * 0.5
        return disc_loss_Y

    def get_lr_scheduler(self, optimizer):
        def lr_lambda(epoch):
            val = 1.0 - max(0, epoch-self.decay_epochs+1.0)/(self.num_epochs-self.decay_epochs+1.0)
            return max(0.0, val)
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    def configure_optimizers(self):
        params = {
            "lr": self.lr,
            "betas": self.betas,
        }
        opt_gen_BA = torch.optim.Adam(self.gen_BA.parameters(), **params)
        opt_gen_AB = torch.optim.Adam(self.gen_AB.parameters(), **params)
        opt_disc_A = torch.optim.Adam(self.disc_A.parameters(), **params)
        opt_disc_B = torch.optim.Adam(self.disc_B.parameters(), **params)
        optimizers = [opt_gen_BA, opt_gen_AB, opt_disc_A, opt_disc_B]
        schedulers = [self.get_lr_scheduler(opt) for opt in optimizers]
        return optimizers, schedulers

    def training_step(self, batch, batch_idx, optimizer_idx):
        real_A = batch["a"]
        real_B = batch["b"]
        if optimizer_idx == 0:
            gen_loss_BA = self.gen_loss(real_B, real_A,
                                        self.gen_BA, self.gen_AB, self.disc_A)
            return gen_loss_BA

        if optimizer_idx == 1:
            gen_loss_AB = self.gen_loss(real_A, real_B,
                                        self.gen_AB, self.gen_BA, self.disc_B)
            return gen_loss_AB

        if optimizer_idx == 2:
            with torch.no_grad():
                fake_A = self.gen_BA(real_B)
            disc_loss_A = self.disc_loss(real_A, fake_A,
                                         self.disc_A)
            return disc_loss_A

        if optimizer_idx == 3:
            with torch.no_grad():
                fake_B = self.gen_AB(real_A)
            disc_loss_B = self.disc_loss(real_B, fake_B,
                                         self.disc_B)
            return disc_loss_B

    def training_epoch_end(self, outputs):
        current_epoch = self.current_epoch + 1

        lr = self.lr_schedulers()[0].get_last_lr()[0]
        self.lr_history.append(lr)
        losses = {}
        for j in range(4):
            loss = [out[j]["loss"].item() for out in outputs]
            self.loss_history[self.loss_names[j]].extend(loss)
            losses[self.loss_names[j]] = np.mean(loss)
        print(
            " - ".join([
                f"Epoch {current_epoch}",
                f"lr: {self.lr_history[-2]:.5f}",
                *[f"{loss}: {val:.5f}" for loss, val in losses.items()],
            ])
        )


        if current_epoch%self.display_epochs==0 or current_epoch in [1, self.num_epochs]:
            os.makedirs("model", exist_ok=True)
            torch.save(self.gen_BA, "model/"+str(current_epoch) + 'amodel.pth')
            torch.save(self.gen_AB, "model/"+str(current_epoch) + 'bmodel.pth')
            torch.set_grad_enabled(False)
            self.eval()
            gen_a = self.forward(self.b_samples)
            show_img(
                torch.cat([self.b_samples, gen_a]),
                nrow=len(self.b_samples),
                title=f"Epoch {current_epoch}: b-to-a Translation",
            )
            torch.set_grad_enabled(True)
            self.train()


    def predict_step(self, batch, batch_idx):
        return self.forward(batch)


    def lr_plot(self):
        num_epochs = len(self.lr_history[:-1])
        plt.figure(figsize=(18, 4.5))
        plt.title("Learning Rate Schedule")
        plt.ylabel("Learning Rate")
        plt.xlabel("Epoch")
        plt.plot(
            np.arange(1, num_epochs+1),
            self.lr_history[:-1],
        )

    def loss_plot(self):
        titles = ["Generator Loss Curves", "Discriminator Loss Curves"]
        num_steps = len(list(self.loss_history.values())[0])
        plt.figure(figsize=(18, 4.5))
        for j in range(4):
            if j%2 == 0:
                plt.subplot(1, 2, (j//2)+1)
                plt.title(titles[j//2])
                plt.ylabel("Loss")
                plt.xlabel("Step")
            plt.plot(
                np.arange(1, num_steps+1),
                self.loss_history[self.loss_names[j]],
                label=self.loss_names[j],
            )
            plt.legend(loc="upper right")

TRAIN_CONFIG = {
    "accelerator": "gpu" if not DEBUG else "cpu",
    "devices": 1,
    "logger": False,
    "enable_checkpointing": True,
    "max_epochs": NUM_EPOCHS,
    "precision": 16 if not DEBUG else 32,
}

dm = CustomDataModule()
model = CycleGAN()
print(model)
model.gen_BA=torch.load("30amodel_sp.pth")
trainer = L.Trainer(**TRAIN_CONFIG)

# trainer.fit(model, datamodule=dm)
#
# model.lr_plot()
#
# model.loss_plot()

predictions = trainer.predict(model, datamodule=dm)

os.makedirs("images", exist_ok=True)

idx = 0
for tensor in predictions:
    for a in tensor:
        save_image((a.float().squeeze()*0.5+0.5), fp=f"../images/{idx}.jpg")
        idx += 1

# shutil.make_archive("/kaggle/working/images", "zip", "/kaggle/images")

torch.set_grad_enabled(False)
model.eval()
#
# for j, bs in enumerate(iter(predict_loader)):
#     if j == 5:
#         break
#     gen_a = model(bs)
#     show_img(
#         torch.cat([bs, gen_a]),
#         nrow=len(bs),
#         title=f"Sample {j+1}: b-to-a Translation",
#     )