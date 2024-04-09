import re

import matplotlib.pyplot as plt
import numpy as np
import os

def get_index_list(nums):
    indices = {}
    result = []

    for i, num in enumerate(nums):
        if num not in indices:
            indices[num] = i

    for num in sorted(indices.keys()):
        result.append(indices[num])

    return sorted(result)

def get_value_by_indices(li,indices):
    res=[]
    for i,value in enumerate(li):
         if i in indices:
             res.append(value)
    return res

def get_aver_loss(epoch_list,loss_list):
    indices=get_index_list(epoch_list)
    start=indices[0]
    end=indices[1]
    res=[]
    for i in range(2,len(indices)+1):
        aver=np.sum(loss_list[start:end])/(end-start)
        res.append(round(aver,4))
        if i==len(indices):
            res.append(np.sum(loss_list[end:])/(len(loss_list)-end))
            break
        start,end=end,indices[i]
    return res

def plot_loss(loss_list,label,sav_dir):
    plt.figure()
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.grid()

    plt.plot(np.arange(1, len(loss_list)+1),loss_list,color="#0080FF")

    plt.title(label,loc='center',family='Times New Roman', weight='heavy', size=20, color='gray')
    plt.xlabel("epochs",family='Times New Roman', weight='heavy', size=16, color='gray',labelpad=8)
    plt.ylabel(label, family='Times New Roman', weight='heavy', size=16, color='gray',labelpad=8)
    os.makedirs("plot/"+sav_dir,exist_ok=True)
    plt.savefig(os.path.join("plot/"+sav_dir,label+".png"))

def plot_all(log_dir,save_dir):
    loss_log_path="checkpoints/"+log_dir+"/loss_log.txt"
    content=""
    with open(loss_log_path, "r", encoding="utf-8") as file:
        content = file.read()
    epoch=re.findall(r'(?<=epoch: )(.*?)(?=,)', content)
    # 从训练时的loss_log.txt中提取各loss(带系数)
    loss_G_A =list(map(float,re.findall(r'(?<=G_A: )(.*?)(?= )', content)))
    loss_G_B=list(map(float,re.findall(r'(?<=G_B: )(.*?)(?= )', content)))
    loss_D_A=list(map(float,re.findall(r'(?<=D_A: )(.*?)(?= )', content)))
    loss_D_B=list(map(float,re.findall(r'(?<=D_B: )(.*?)(?= )', content)))
    loss_idt_A =list(map(float, re.findall(r'(?<=idt_A: )(.*?)(?= )', content)))
    loss_idt_B=list(map(float,re.findall(r'(?<=idt_B: )(.*?)(?=\n|$)', content)))
    loss_cycle_A=list(map(float,re.findall(r'(?<=cycle_A: )(.*?)(?= )', content)))
    loss_cycle_B=list(map(float,re.findall(r'(?<=cycle_B: )(.*?)(?= )', content)))
    # 计算loss
    loss_G = [round(x1+x2+x3+x4+x5+x6,4) for x1, x2, x3,x4,x5,x6 in zip(loss_G_A,loss_G_B,loss_idt_A,loss_idt_B,loss_cycle_A,loss_cycle_B)]
    loss_G_GAN=[round(x1+x2,4) for x1, x2 in zip(loss_G_A,loss_G_B)]
    loss_G_cycle=[round(x1+x2,4) for x1, x2 in zip(loss_cycle_A,loss_cycle_B)]
    loss_G_idt=[round(x1+x2,4) for x1, x2 in zip(loss_idt_A,loss_idt_B)]
    loss_D=[round(x1+x2,4) for x1, x2 in zip(loss_D_A,loss_D_B)]
    # 绘制loss，相同epoch内的不同iters取平均值，每个epoch为一个点
    plot_loss(get_aver_loss(epoch,loss_G),"loss_G",save_dir)
    plot_loss(get_aver_loss(epoch,loss_G_GAN),"loss_G_GAN",save_dir)
    plot_loss(get_aver_loss(epoch,loss_G_cycle),"loss_G_cycle",save_dir)
    plot_loss(get_aver_loss(epoch,loss_G_idt),"loss_G_idt",save_dir)
    plot_loss(get_aver_loss(epoch,loss_D),"loss_D",save_dir)


if __name__ == '__main__':
    # plot_all("shuimo_cyclegan", "map_unet")
    plot_all("map_resnet", "map_resnet")