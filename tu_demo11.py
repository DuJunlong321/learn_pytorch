# -*- coding: UTF-8 -*-
"""===============================================
@Author ：XiaoYuan Max
@Date   ：2022/10/28
@IDE    ：PyCharm
@File   ：小土堆pytorch 现有模型的使用及修改 模型的保存和读取
@intro  ：
=================================================="""
import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential

"""
demo1：加载vgg训练好的模型，并在里边加入一个线性层
"""
# ImageNet数据集太大了，100多G，还是用CIFAR10吧
# train_data = torchvision.datasets.ImageNet("../data_image_net", split='train', download=True,
#                                         transform=torchvision.transforms.ToTensor())
train_data = torchvision.datasets.CIFAR10('../data', train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
# 加载现有的vgg模型
vgg16_not_pretrain = torchvision.models.vgg16(pretrained=False)
vgg16_pretrained = torchvision.models.vgg16(pretrained=True)

# 修改方法1：加入一个线性层,编号7
vgg16_pretrained.add_module("7", nn.Linear(1000, 10))
print(vgg16_pretrained)

# 修改方法2：修改原来的第六个线性层
vgg16_not_pretrain.classifier[6] = nn.Linear(4096, 10)
print(vgg16_not_pretrain)


"""
demo2:保存/加载模型
两种保存方法 对应两种加载方法
保存模型都是用torch.save，加载模型都是用torch.load，一起保存的时候save整个模型，加载时直接torch.load加载
保存时只保存参数的，需要先向model vgg加载结构，再用model vgg.load state dict加载参数，加载参数还是要torc.load方法
保存方法1的‘陷阱’：
在使用方法1保存现有模型时，不会出错，代码更少，但是使用方法1保存自己的模型时，必须要引入这个模型的定义才可以
"""
# 保存东西需要现有东西保存
vgg16 = torchvision.models.vgg16(pretrained=False)
# 保存方式1：模型结构+参数一起保存
torch.save(vgg16, "vgg16_pretrained_save_method1.pth")
# 多保存一个vgg16_pretrained，后面 完整模型测试讨论会用到
torch.save(vgg16_pretrained, "vgg16_pretrained_save_method1.pth")
# 加载方式1
model1 = torch.load("vgg16_save_method1.pth")


# 保存方式2：只保存模型参数
torch.save(vgg16.state_dict(), "vgg16_save_method2.pth")
# 加载方式2
# 先加载模型结构
model_vgg = torchvision.models.vgg16(pretrained=False)
# 再加载模型参数
model_vgg.load_state_dict(torch.load("vgg16_save_method2.pth"))


# 保存方法1的‘陷阱’
"""先保存tudui模型
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x

注意，保存的是tudui实例，而不是Tudui类
tudui = Tudui()
torch.save(tudui, "tudui_save_method1.pth")
"""

"""
直接加载tudui模型会报错
tudui = torch.load("tudui_save_method1.pth")
报错：
AttributeError: Can't get attribute 'Tudui' ...>
"""

"""
需要先把网络结构放进来，import 或者把class的定义代码粘贴过来
这里要把tu demo10中训练的代码注释掉，不然运行到class类之后，会继续往下执行训练的代码
虽然from代码没有高亮 但是如果删掉 程序就会报错
"""
from tu_demo10 import Tudui
tudui = torch.load("tudui_save_method1.pth")
print("Successfully load model!")

