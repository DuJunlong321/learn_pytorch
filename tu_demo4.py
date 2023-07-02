# -*- coding: UTF-8 -*-
"""===============================================
@Author ：XiaoYuan Max
@Date   ：2022/10/27
@IDE    ：PyCharm
@File   ：小土堆pytorch 常见的transforms
@intro  ：transform是一个py文件，其中tosensor compose normalize是很常用的操作
=================================================="""
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

'''
demo1:魔法函数
实例化一个对象之后，如果这个对象可以不用".方法名"，而是可以直接传参数，那说明这个对象就有一个对应的魔法函数
比如getitem call len
'''
img_path = "dataset/train/ants/0013035.jpg"
img = Image.open(img_path)
writer = SummaryWriter('logs')

'''
demo2:compose的用法
compose就是一个指定一个transform操作序列，定义了一条加工流水线
Example:
    >>> transforms.Compose([
    >>>     transforms.CenterCrop(10),
    >>>     transforms.PILToTensor(),
    >>>     transforms.ConvertImageDtype(torch.float),
    >>> ])
'''

'''
demo3:normalize的用法
计算方法：output[channel] = (input[channel] - mean[channel]) / std[channel]
说人话：该像素上的值减去均值，再除以方差
'''
trans_norm = transforms.Normalize([5, 0.5, 0.5], [0.5, 0.5, 0.5])
trans_tensor = transforms.ToTensor()
img_tensor = trans_tensor(img)
img_norm = trans_norm(img_tensor)

writer.add_image('norm img', img_norm)
# 在board中可以看到norm之后图片变化很大，因为故意设的均值很大

'''
demo4:resize的用法
Resize the input image to the given size.
注意如果给了一个int就是变为正方形，给（H，W）才是H W
resize不会改变图片的数据类型
注意区别:
transforms.Resize 修改 PIL
torch.reshape     修改 tensor
'''
trans_resize = transforms.Resize((512, 500))
img_resize = trans_resize(img)
img_resize = trans_tensor(img_resize)
print(img_resize)

'''
demo5:random crop
随机剪一个指定尺寸的新图片
在随机位置裁剪给定图像。
'''
trans_randomcrop = transforms.RandomCrop(500)
trans_compose = transforms.Compose([
    trans_randomcrop,
    trans_tensor
])
for i in range(10):
    img_crop_tensor = trans_compose(img)
    print(img_crop_tensor.shape)
    writer.add_image('crop tensor', img_crop_tensor, i)

writer.close()
