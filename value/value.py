
import os
import shutil
import torchvision.transforms as transforms
from PIL import Image
import re
from psnr import get_mean_PSNR
from ssim import get_mean_SSIM

def split(folder_path):
    """
    将 test.py 程序生成的 result/[model_name] 文件夹下分为
    A_real(test.py输入的测试图片)
    B_fake(test.py生成的图片）
    Args:
        folder_path:

    Returns:

    """
    all_files = os.listdir(folder_path)
    direction="A" if "A" in all_files[0] else "B"
    folder_real= os.path.join(folder_path,"A_real")
    folder_fake = os.path.join(folder_path,"B_fake")
    os.makedirs(folder_real, exist_ok=True)
    os.makedirs(folder_fake, exist_ok=True)
    for file_name in all_files:
        file_path = os.path.join(folder_path, file_name)
        if "real" in file_name:
            shutil.move(file_path, os.path.join(folder_real, file_name))
        else:
            shutil.move(file_path, os.path.join(folder_fake, file_name))

def copy(source_folder, destination_folder, num_files):
    """
    复制文件夹
    Args:
        source_folder:
        destination_folder:
        num_files:

    Returns:

    """
    all_files = os.listdir(source_folder).sort()
    os.makedirs(destination_folder, exist_ok=True)
    for i, file_name in enumerate(all_files):
        if i >= num_files:
            break
        source_path = os.path.join(source_folder, file_name)
        destination_path = os.path.join(destination_folder, file_name)
        shutil.copy(source_path, destination_path)

def transform(img):
    """
    处理图像方便比较
    Args:
        img:

    Returns:

    """
    transform = transforms.Compose([
        transforms.Resize(286, Image.BICUBIC),
        transforms.RandomCrop(256),
        transforms.RandomHorizontalFlip(),
    ])
    transformed_image = transform(img)
    return transformed_image

def transfrom_and_save(source_folder, destination_folder, num_files):
    """
    处理图像并保存到新文件夹
    Args:
        img:

    Returns:

    """
    all_files = os.listdir(source_folder)
    all_files=sorted(all_files)
    os.makedirs(destination_folder, exist_ok=True)
    for i, file_name in enumerate(all_files):
        if i >= num_files:
            break
        source_path = os.path.join(source_folder, file_name)
        destination_path = os.path.join(destination_folder, file_name)
        img=Image.open(source_path)
        img_=transform(img)
        img_.save(destination_path)

def count_files_in_folder(folder_path):
    return len([name for name in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, name))])

def crossline():
    print("--------------------------------------------------")

def value(results_dir,datasets_dir,transform=False,direction="both"):

    print("Start to value results")


    split("../results/"+results_dir+"/test_latest/images") # 分割结果文件夹
    print("Split test results folder into fake and real folders")

    if transform: # 处理图像
        transfrom_and_save(
            "../datasets/"+datasets_dir+"/testB",
            "../results/"+results_dir+"/test_latest/images/B_real",
            count_files_in_folder("../results/"+results_dir+"/test_latest/images/B_fake"))
    else:
        source_folder_path ="../datasets/"+datasets_dir+"/testB"
        destination_folder_path = "../results/"+results_dir+"/test_latest/images/B_real"
        num_files_to_copy = count_files_in_folder("../results/"+results_dir+"/test_latest/images/B_fake")  # 指定要复制的文件数量
        copy(source_folder_path, destination_folder_path, num_files_to_copy)

    print("Copy and transform real images (domain B)")

    # 计算FID，PSNR，SSIM
    print("Calculate FID score...")
    cmd="python -m pytorch_fid ../results/"+\
          results_dir+\
          "/test_latest/images/B_real " \
          "../results/"+ \
          results_dir + \
          "/test_latest/images/B_fake" \
          "  --device cuda:0"
    output_lines = os.popen(cmd).readlines()
    pattern = r"FID:\s*(\d+\.\d+)"
    matches = re.findall(pattern, str(output_lines))
    FID=float(str(matches[0]))

    B_real_dir="../results/"+results_dir+"/test_latest/images/B_real"
    B_fake_dir="../results/"+results_dir+"/test_latest/images/B_fake"

    # PSNR计算方式：B_real和B_fake文件夹随机图片之间计算psnr并取平均值，重复以上过程取平均值
    print("Calculate PSNR score...")
    PSNR=round(get_mean_PSNR(B_fake_dir,B_real_dir,100),4)

    # SSIM计算方式：B_real和B_fake文件夹随机图片之间计算ssim并取平均值，重复以上过程取平均值
    print("Calculate SSIM score...")
    SSIM=get_mean_SSIM(B_fake_dir,B_real_dir,10)

    crossline()
    print("FID : "+str(round(FID,4)))
    print("PSNR : " + str(round(PSNR,4)))
    print("SSIM : " + str(round(SSIM,4)))
    crossline()

    str_to_save = "FID : "+str(round(FID,4))+"\n"+"PSNR : " + str(round(PSNR,4))+"\n"+"SSIM : " + str(round(SSIM,4))

    with open("../results/"+results_dir+"/value.txt", "w", encoding="utf-8") as f:
        f.write(str_to_save)




if __name__ == "__main__":
    # value("shuimo_unet","shuimo",transform=True,dircetion="both")
    # value("shuimo_resnet","shuimo",transform=True,dircetion="both")
    # value("map_resnet", "maps", transform=True)
    # value("map_unet", "maps", transform=True)
    value("painting_resnet", "vangogh2photo", transform=True)
    # value("painting_unet", "vangogh2photo", transform=True)
    # value("painting_saunet", "vangogh2photo", transform=True)
    # value("painting_saunet_old", "vangogh2photo", transform=True)



