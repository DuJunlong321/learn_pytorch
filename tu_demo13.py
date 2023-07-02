# -*- coding: UTF-8 -*-
"""===============================================
@Author ：XiaoYuan Max
@Date   ：2022/10/28
@IDE    ：PyCharm
@File   ：小土堆pytorch 利用GPU训练
@intro  ：使用gpu训练，能提高10倍训练速度
=================================================="""
"""
demo1:在特定位置加入.cuda()
能加的有3个地方：模型 loss 模型输入
    tudui = tudui.cuda()
    loss_fn = loss_fn.cuda()
    imgs, targets = data
    imgs = imgs.cuda()
    targets = targets.cuda()
    
先判断再cuda
if torch.cuda_is_available():
    tudui = tudui.cuda()
    loss_fn = loss_fn.cuda()
    imgs, targets = data
    imgs = imgs.cuda()
    targets = targets.cuda()
"""


"""
demo2:在特定位置加入.to(device)
只有cpu
device = torch.device("cpu")
只有一张显卡
device = torch.device("cuda")
device = torch.device("cuda:0")
有多张显卡
device = torch.device("cuda:0")
device = torch.device("cuda:1")
能加的有3个地方：模型   loss   数据
    tudui = tudui.to(device)
    loss_fn = loss_fn.to(device)
    imgs, targets = data
    imgs = imgs.to(device)
    targets = targets.to(device)
先判断再to
device = torch.device("cuda", if torch.cuda_is_available() else "cpu")
"""

