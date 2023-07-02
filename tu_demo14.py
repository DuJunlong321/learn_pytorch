# -*- coding: UTF-8 -*-
"""===============================================
@Author ：XiaoYuan Max
@Date   ：2022/10/28
@IDE    ：PyCharm
@File   ：小土堆pytorch 完整的模型测试套路
@intro  ：利用已经训练好的模型进行测试，我这里用了训练好的vgg16，和土哥代码不一样
=================================================="""
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
# from tu_demo12 import Tudui
"""
demo1:找一个图片让vgg model去识别是哪个类
代码解释：
image = image.convert('RGB')
由于图片有png jpg的不同格式，而png图片是四通道的 多一个透明度通道，jpg是三通道的 只有三个颜色通道
这一行代码可以让png jpg都只有三个颜色通道，增强了代码的适应性

demo14.py对应土堆的test.py 但是我这里加载的是vgg16 分类1000的模型 所以就不需要class的代码了
因为加class的代码是为了能按照加载方式1加载tudui模型

map_location=torch.device('cpu')
这个pth文件，如果是gpu训练得来的，就要放到gpu下运行，如果你想在cpu下运行，就加上这个代码
"""
# # 加载图片
image_path = "./img/img.png"
image = Image.open(image_path)
image = image.convert('RGB')
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])
image = transform(image)
print(image.shape)


# 这里有个小疑问，明明模型是修改了的，加了线性层分类10类的，但是output还是1000个类别的概率
# 模型要四维带batch的图片类型，上边image.shape看到image是三维的
image = torch.reshape(image, (1, 3, 32, 32))
print(image.shape)
image = image.cuda()

# 加载模型,查看模型输出
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

model = torch.load("tudui_0.pth")
print(model)
model.eval()
with torch.no_grad():
    output = model(image)
print(output)

# 找到最大的概率值的位置 查看数字对应类别在debug datasets.CIFAR10的class to idx
print(output.argmax(1))


