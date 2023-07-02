# -*- coding: UTF-8 -*-
"""===============================================
@Author ：XiaoYuan Max
@Date   ：2022/10/27
@IDE    ：PyCharm
@File   ：小土堆pytorch P7 TensorBoard的使用
@intro  ：TensorBoard学习的理由有2
          1.可视化transform的输出，为了讲好理解好transform
          2.今后分析loss，分析模型阶段的输出很有用
=================================================="""
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tu_demo1 import MyData

'''
demo1:SummaryWriter的介绍
Writes entries directly to event files in the log_dir to be consumed by TensorBoard. 
SummaryWriter把实体直接写入一个目录里边，这个目录可以被TensorBoard读取，然后画成图
SummaryWriter常用只有1个参数log_dir
Example：
    传入log dir参数
    writer = SummaryWriter("my_experiment")
    # folder location: my_experiment，比如后边常写的logs
    comment参数可以给log dir后边加后缀
    writer = SummaryWriter(comment="LR_0.1_BATCH_16")
    # folder location: runs/May04_22-14-54_s-MacBook-Pro.localLR_0.1_BATCH_16/
    #                  runs/May04_22-14-54_s-MacBook-Pro.local是默认的目录
'''
# 使用步骤:1.实例化 2.往里写数据 3.关闭
# writer = SummaryWriter("logs")

# writer.add_image()
# writer.add_images()
# writer.add_scalar()

# writer.close()


'''
demo2：写标量数据
tag (string): Data identifier 标识符，表名
scalar_value (float or string/blobname): Value to save  y轴
global_step (int): Global step value to record          x轴
'''
# writer = SummaryWriter("logs")
#
# for i in range(100):
#     writer.add_scalar("y=x summary", i, i)
#     writer.add_scalar("y=2x summary", 2*i, i)
#
# writer.close()


'''
demo3：写图片数据
tag (string): Data identifier
img_tensor (torch.Tensor, numpy.array, or string/blobname): Image data
global_step (int): Global step value to record

1.使用numpy.array()
使用numpy.array()会报错，因为你的图片维度顺序不符合writer的要求，报错如下：
TypeError: Cannot handle this data type: (1, 1, 512)
改正如下，使用dataformats='HWC'向writer说明我们的三位顺序是HWC：
writer.add_image('my_image_HWC', img_HWC, 0, dataformats='HWC')
可以在控制台简单运行一下代码，就知道顺序了

2.使用Transform.transforms.Totensor()
写图片最简单的是:用Transform的Totensor()方法,将PIL转换成Totensor方法
transforms = Transform.transforms.Totensor()
img_trans = transforms(img_PIL)
'''
# writer = SummaryWriter("logs")
#
# image_path = "dataset/train/ants/0013035.jpg"
# img_pil = Image.open(image_path)
# img_array = np.array(img_pil)
#
# # writer.add_image("img test", img_array, 1)
# writer.add_image("img test", img_array, 1, dataformats='HWC')
# writer.close()

'''
demo4:练习使用图片写入
'''
# 获取数据集
root_dir = "./dataset/train"
label_dir = "ants"
ants_dataset = MyData(root_dir, label_dir)
# 写入board
step = 1
writer = SummaryWriter("logs")
for img, label in ants_dataset:
    img_array = np.array(img)
    print(type(img_array), img_array.shape)
    print("正在写入第{}张图片".format(step))
    writer.add_image("ants images", img_array, step, dataformats='HWC')
    step = step + 1
# 最后一定关闭writer
writer.close()
# 第117张图片的shape是（300，300）没有第三维数据，写不进去，会报错，但是board能存116个
# 有解决办法的欢迎联系我
