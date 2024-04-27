import os
def test_all():
    # A painting B photo

    cmd1 = "python test.py --dataroot datasets/vangogh2photo/testB --name painting_resnet_b2a --model test --no_dropout"
    cmd2 = "python test.py --dataroot datasets/vangogh2photo/testB --name painting_unet_b2a --model test --no_dropout --netG unet_256"
    cmd3 = "python test.py --dataroot datasets/vangogh2photo/testB --name painting_saunet_b2a --model test --no_dropout --netG saunet_256 --netD saunet"

    cmd4 = "python test.py --dataroot datasets/vangogh2photo/testA --name painting_resnet_a2b --model test --no_dropout"
    cmd5 = "python test.py --dataroot datasets/vangogh2photo/testA --name painting_unet_a2b --model test --no_dropout --netG unet_256"
    cmd6 = "python test.py --dataroot datasets/vangogh2photo/testA --name painting_saunet_a2b --model test --no_dropout --netG saunet_256 --netD saunet"

    # cmd1 = "python test.py --dataroot datasets/vangogh2photo/testB --name painting_resnet_b2a --model test --no_dropout"
    # cmd2 = "python test.py --dataroot datasets/vangogh2photo/testB --name painting_unet_b2a --model test --no_dropout --netG unet_256"
    # cmd3 = "python test.py --dataroot datasets/vangogh2photo/testB --name painting_saunet_b2a --model test --no_dropout --netG saunet_256 --netD saunet"
    #
    # cmd4 = "python test.py --dataroot datasets/vangogh2photo/testA --name painting_resnet_a2b --model test --no_dropout"
    # cmd5 = "python test.py --dataroot datasets/vangogh2photo/testA --name painting_unet_a2b --model test --no_dropout --netG unet_256"
    # cmd6 = "python test.py --dataroot datasets/vangogh2photo/testA --name painting_saunet_a2b --model test --no_dropout --netG saunet_256 --netD saunet"

    output_lines = os.popen(cmd1).readlines()
    print(output_lines)
    # output_lines = os.popen(cmd2).readlines()
    # print(output_lines)
    # output_lines = os.popen(cmd3).readlines()
    # print(output_lines)
    # output_lines = os.popen(cmd4).readlines()
    # output_lines = os.popen(cmd5).readlines()
    # output_lines = os.popen(cmd6).readlines()

if __name__ == '__main__':
    test_all()
