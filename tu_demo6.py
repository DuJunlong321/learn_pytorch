# -*- coding: UTF-8 -*-
"""===============================================
@Author ：XiaoYuan Max
@Date   ：2022/10/27
@IDE    ：PyCharm
@File   ：小土堆pytorch torchvision中dataloader的使用
@intro  ：loader意思是加载器，就像坦克的装弹手，在把数据放进网络之前，就用loader装弹
=================================================="""
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

'''
demo1:dataloader的简单使用
batch size：loader能每次装弹4枚进入枪膛，或者理解每次抓四张牌
shuffle：每次epoch是否打乱原来的顺序，就像打完一轮牌后，洗不洗牌
drop last：最后的打他不够一个batch 还要不要了
'''
train_set = torchvision.datasets.CIFAR10("./dataset", train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_set = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(), download=True)

train_loader = DataLoader(dataset=train_set, batch_size=4, shuffle=True, num_workers=0, drop_last=False)
test_loader = DataLoader(dataset=test_set, batch_size=4, shuffle=True, num_workers=0, drop_last=False)

img, target = test_set[0]
print(img.shape)
print(target)
# 使用board可视化
writer = SummaryWriter("logs")
step = 0
for data in test_loader:
    imgs, targets = data
    # print(imgs.shape)
    # print(targets)

    # 这里如果你用add image会报错，因为这个方法只能一次写一个图片，你必须换add images方法来写入带有批处理的图片
    # writer.add_image("test set loader", imgs, step)
    writer.add_images("test set loader", imgs, step)
    step = step + 1

writer.close()