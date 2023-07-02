# -*- coding: UTF-8 -*-
"""===============================================
@Author ：XiaoYuan Max
@Date   ：2022/10/27
@IDE    ：PyCharm
@File   ：小土堆pytorch 神经网络 基本骨架
@intro  ：
=================================================="""
import torch
from torch import nn

"""
demo1:写一个最基本的神经网络都要哪些骨头
两个骨头就是骨架：__init__ forward
"""
class Tudui(nn.module):
    def __init__(self):
        super(Tudui, self).__init__()

    def forward(self, input):
        output = input + 1
        return output


tudui = Tudui()
x = torch.tensor(1.0)
output = tudui(x)
print(output)
