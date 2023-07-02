# -*- coding: UTF-8 -*-
"""===============================================
@Author ：XiaoYuan Max
@Date   ：2022/10/28
@IDE    ：PyCharm
@File   ：小土堆pytorch 完整的模型训练套路
@intro  ：
=================================================="""
import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

"""
demo1:完整的模型训练测试流程
细节代码解释：
outputs.argmax(1)：outputs是一个元素是列表的列表，即二维矩阵，64行10列，argmax(1)是令行方向上的最大值为1，其余为0
with torch.no_grad():测试前要把梯度关闭
tudui.train() tudui.eval()：训练前 测试前写一下
"""

# 选择gpu 或 cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data = torchvision.datasets.CIFAR10("./dataset", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)

# 看一看训练集 测试集的长度
train_data_len = len(train_data)
test_data_len = len(test_data)
print("训练集的长度是：{}".format(train_data_len))
print("测试集的长度是：{}".format(test_data_len))

# 使用dataloader加载数据
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)


# 创建网络模型
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, 1, 2),
            MaxPool2d(2),
            Conv2d(32, 32,5, 1, 2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, 1, 2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x


# 创建模型
tudui = Tudui()
# 上gpu
tudui = tudui.to(device)
# 损失函数 优化器
loss_fn = nn.CrossEntropyLoss()
# 上gpu
loss_fn = loss_fn.to(device)
learing_rate = 0.01
optimizer = torch.optim.SGD(tudui.parameters(), lr=learing_rate)

# 设置控制训练次数的参数
# 记录训练 测试次数
total_train_step = 0
total_test_step = 0
# 训练轮数
epochs = 1
# 写入board
writer = SummaryWriter("logs")

for epoch in range(epochs):
    print("-------第 {} 轮训练开始-------".format(epoch + 1))

    # 训练步骤开始
    tudui.train()
    for data in train_dataloader:
        imgs, targets = data
        # 上gpu
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = tudui(imgs)
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        # batch=64,训练集=5W，学习一遍训练集就需要50000/64=781.25次训练
        total_train_step += 1
        writer.add_scalar("train loss", loss.item(), total_train_step)
        if total_train_step % 100 == 0:
            print("训练次数：{}， loss：{}".format(total_train_step, loss.item()))


    # 测试步骤开始
    tudui.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            # 上gpu
            imgs = imgs.to(device)
            targets = targets.to(device)
            # print(targets)
            outputs = tudui(imgs)
            # print(outputs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy

    print("整体测试集上的Loss: {}".format(total_test_loss))
    print("整体测试集上的正确率: {}".format(total_accuracy / test_data_len))
    writer.add_scalar("test loss", total_test_loss, total_test_step)
    writer.add_scalar("test accuracy", total_accuracy/test_data_len, total_test_step)
    total_test_step = total_test_step + 1

    torch.save(tudui, "tudui_{}.pth".format(epoch))
    print("模型已保存")

writer.close()
