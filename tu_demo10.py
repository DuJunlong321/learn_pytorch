# -*- coding: UTF-8 -*-
"""===============================================
@Author ：XiaoYuan Max
@Date   ：2022/10/27
@IDE    ：PyCharm
@File   ：小土堆pytorch 神经网络 损失函数 优化器
@intro  ：介绍如何使用loss 优化器 哪些值得注意的代码行
=================================================="""
import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

"""
demo1:在模型训练中加入损失函数和优化器
"""
dataset = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)

dataloader = DataLoader(dataset, batch_size=1)

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

# 这里注释是因为后边我保存/加载模型那一节的原因，在学习loss optim这一部分时可以取消注释
# tudui = Tudui()
# # 损失函数 优化器
# loss = nn.CrossEntropyLoss()
# optim = torch.optim.SGD(tudui.parameters(), lr=0.01)
#
# for i in range(20):
#     running_loss = 0.0
#     for data in dataloader:
#         imgs, targets = data
#         outputs = tudui(imgs)
#         result_loss = loss(outputs, targets)
#         # 注意 清零--》反向传播算梯度--》更新参数
#         optim.zero_grad()
#         result_loss.backward()
#         optim.step()
#         running_loss = running_loss + result_loss
#     print(running_loss)

# running_loss：
# tensor(18726.5977, grad_fn=<AddBackward0>)
# tensor(16132.8926, grad_fn=<AddBackward0>)
# tensor(15426.6357, grad_fn=<AddBackward0>)
