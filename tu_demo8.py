# -*- coding: UTF-8 -*-
"""===============================================
@Author ：XiaoYuan Max
@Date   ：2022/10/27
@IDE    ：PyCharm
@File   ：小土堆pytorch 神经网络 卷积操作
@intro  ：理解什么是卷积操作 怎么算卷积结果
=================================================="""
import torch
import torch.nn.functional as F

"""
demo1:使用conv2d
input kernel:都是四维
stride：步长
padding：如果步长是1，又想保持输入输出高宽不变，就把padding设置1
"""
input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]])

kernel = torch.tensor([[1, 2, 1],
                       [0, 1, 0],
                       [2, 1, 0]])

# conv2d需要输入的tensor是四维的（batch， c，h，w），但是现在的input kernel是二维
input = torch.reshape(input, (1, 1, 5, 5))
kernel = torch.reshape(kernel, (1, 1, 3, 3))

output = F.conv2d(input, kernel, stride=1)
print(output)
# tensor([[[[10, 12, 12],
#           [18, 16, 16],
#           [13,  9,  3]]]])

output2 = F.conv2d(input, kernel, stride=1, padding=1)
print(output2)
# tensor([[[[ 1,  3,  4, 10,  8],
#           [ 5, 10, 12, 12,  6],
#           [ 7, 18, 16, 16,  8],
#           [11, 13,  9,  3,  4],
#           [14, 13,  9,  7,  4]]]])
