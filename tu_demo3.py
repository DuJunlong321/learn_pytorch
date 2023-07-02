# -*- coding: UTF-8 -*-
"""===============================================
@Author ：XiaoYuan Max
@Date   ：2022/10/27
@IDE    ：PyCharm
@File   ：小土堆pytorch P9 transforms的使用
@intro  ：transform是一个py文件，其中tosensor compose normalize是很常用的操作
=================================================="""
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

'''
demo1:tosensor简单使用 
tosensor：
Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor
example:
    pil --> tensor
    tensor([[[0.3137, 0.3137, 0.3137,  ..., 0.3176, 0.3098, 0.2980],
             [0.3176, 0.3176, 0.3176,  ...,
'''
# 获取pil类型图片
img_path = "dataset/train/ants/0013035.jpg"
img = Image.open(img_path)
# 创建需要的transforms工具，并给工具起名字
tensor_trans = transforms.ToTensor()
# 使用工具
tensor_img = tensor_trans(img)
print(tensor_img)

"""
demo2:为什么需要tensor数据类型
因为tensor包含了一些属性是计算神经网络是必不可少的
grad：梯度
device：设备
is CUDA：
requires grad：保留梯度
"""
# tensor_img.grad = 0
# tensor_img.requires_grad = False

'''
demo3:使用tensor数据类型写入board
'''
writer = SummaryWriter("logs")
writer.add_image('tensor img', tensor_img, 1)
writer.close()
