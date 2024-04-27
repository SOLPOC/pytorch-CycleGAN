import os

# --dataroot datasets/vangogh2photo/testA --name painting_resnet --model test --no_dropout
# --dataroot datasets/vangogh2photo/testA --name painting_unet --model test --no_dropout --netG unet_256
# --dataroot datasets/vangogh2photo/testA --name painting_unet_sp --model test --no_dropout --netG unet_256
# --dataroot datasets/vangogh2photo/testA --name painting_unet_pixel --model test --no_dropout --netG unet_256 --netD pixel
# --dataroot datasets/vangogh2photo/testA --name painting_saunet --model test --no_dropout --netG saunet_256 --netD saunet
# --dataroot datasets/vangogh2photo/testA --name painting_saunet_old --model test --no_dropout --netG saunet_256 --netD saunet
def test_all(models):
      for model in models:
            print("Start to test "+model)
            cmd = "python train
            output_lines = os.popen(cmd).readlines()
            print(output_lines)