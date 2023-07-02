# -*- coding: UTF-8 -*-
"""===============================================
@Author ：XiaoYuan Max
@Date   ：2022/10/27
@IDE    ：PyCharm
@File   ：小土堆pytorch torchvision中数据集dataset的使用
@intro  ：transform和torchvision中数据集的联合使用
=================================================="""
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

'''
demo1:使用torchvision获取数据集
前面是数据的类型，后面是图片的标签 数字对应类别
(<PIL.Image.Image image mode=RGB size=32x32 at 0x22F00A48F10>, 6)
类别：
['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
'''
# train_set = torchvision.datasets.CIFAR10(root="./dataset", train=True, download=True)
# test_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, download=True)
# print(train_set[0])
# print(train_set.classes)
# img, target = train_set[0]
# print(train_set.classes[target])

'''
demo2:使用compose对数据集做transform操作
'''
dataset_trans = transforms.Compose([
    torchvision.transforms.ToTensor()
])
train_set = torchvision.datasets.CIFAR10(root="./dataset", train=True, transform=dataset_trans, download=True)
test_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=dataset_trans, download=True)

writer = SummaryWriter('logs')
for i in range(10):
    img, target = train_set[i]
    writer.add_image('test torchvison compose', img, i)

writer.close()