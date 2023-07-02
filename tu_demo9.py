# -*- coding: UTF-8 -*-
"""===============================================
@Author ：XiaoYuan Max
@Date   ：2022/10/27
@IDE    ：PyCharm
@File   ：小土堆pytorch 神经网络 卷积层 最大池化层 激活函数 线性层 Sequential使用 搭建实战
@intro  ：讲的很细致，分别介绍四个常用计算操作，最后进行一个vgg搭建实战收尾
=================================================="""
import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, ReLU, Sigmoid, Linear, Flatten, Sequential
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

"""
demo1:向神经网络骨架中添加一个卷积层，并可视化查看卷积结果
"""
# 使用测试集，因为比较小
dataset = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset=dataset, batch_size=64)


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x


tudui = Tudui()
print(tudui)
writer = SummaryWriter('logs')

step = 0
for data in dataloader:
    imgs, target = data
    output = tudui(imgs)
    print(imgs.shape)
    print(output.shape)
    # torch.Size([64, 3, 32, 32])
    writer.add_images("before conv2d", imgs, step)
    # torch.Size([64, 6, 30, 30])
    # 因为channel是6，board不知道该怎么写入图片了，所以要reshape
    output = torch.reshape(output, (-1, 3, 30, 30))
    writer.add_images("after conv2d", output, step)
    step = step + 1

writer.close()

"""
demo2:向神经网络骨架中添加一个池化层，并可视化查看池化结果
这一部分代码和上边几乎一模一样，需要注意的是，池化层必须直接作用在float数据类型上，所以如果使用torch.tensor的话
就要加上dtype=float32，然后同样还要reshape为四维tensor
ceil mode：池化核走出input时还要不要里边的最大值 默认不要
"""
class Tudui2(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=False)

    def forward(self, input):
        output = self.maxpool1(input)
        return output


tudui = Tudui2()

writer = SummaryWriter("../logs_maxpool")
step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, step)
    output = tudui(imgs)
    writer.add_images("output", output, step)
    step = step + 1

writer.close()


"""
demo3:向神经网络骨架中添加一个激活函数，并可视化结果
由于代码基本一致，只写网络骨架的代码吧
"""
class Tudui3(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.relu1 = ReLU()
        self.sigmoid1 = Sigmoid()

    def forward(self, input):
        output = self.sigmoid1(input)
        return output


"""
demo4:向神经网络骨架中添加一个线性层，并可视化结果
由于代码基本一致，只写网络骨架的代码吧，只是特别注意这里在把图片放入线性层之前要用flatten把图片弄成一维的
"""
class Tudui4(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.linear1 = Linear(196608, 10)

    def forward(self, input):
        output = self.linear1(input)
        return output


tudui = Tudui()

for data in dataloader:
    imgs, targets = data
    print(imgs.shape)
    # 线性层只能处理一维tensor
    output = torch.flatten(imgs)
    print(output.shape)
    output = tudui(output)
    print(output.shape)


"""
demo5:搭建一个vgg神经网络
"""
class Vgg(nn.Module):
    def __init__(self):
        super(Vgg, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, kernel_size=5, padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(32, 32, kernel_size=5, padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(32, 64, kernel_size=5, padding=2),
            MaxPool2d(kernel_size=2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x


vgg = Vgg()

input = torch.ones((64, 3, 32, 32))
output = vgg(input)
print(output.shape)

writer = SummaryWriter("logs")
writer.add_graph(vgg, input)
writer.close()