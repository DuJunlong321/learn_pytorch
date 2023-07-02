# -*- coding: UTF-8 -*-
"""===============================================
@Author ：XiaoYuan Max
@Date   ：2022/10/27
@IDE    ：PyCharm
@File   ：小土堆pytorch P6 DataSet类代码实战
@intro  ：这节课主要重写了DataSet类的代码，以后如果要处理自己的数据集，就需要重写这个类
=================================================="""
import os
from torch.utils.data import Dataset
import torchvision
from PIL import Image

'''
demo1:使用PIL打开一个图片
windows下所有的绝对路径都要把\换成\\
不然就会报错：
OSError: [Errno 22] Invalid argument: 'D:\\pycharm download\torch\\dataset\train\x07nts\x013035.jpg'
'''
# img_path = "D:\\pycharm download\\torch\\dataset\\train\\ants\\0013035.jpg"
# img = Image.open(img_path)
# img.show()

'''
# demo2:使用os.listdir打开一列图片
# 输出：
# ['0013035.jpg', '1030023514_aad5c608f9.jpg', '1095476100_3906d8afde.jpg', '1099452230_d1949d3250.jpg', '116570827_e9c126745d.jpg', '1225872729_6f0856588f.jpg', '1262877379_64fcada201.jpg', '1269756697_0bce92cdab.jpg', '1286984635_5119e80de1.jpg',
# '''
# dir_path = "dataset/train/ants"
# img_path_list = os.listdir(dir_path)
# print(img_path_list)


'''
demo3：实现自己的dataset类
继承的类，必须实现父类的所有接口/重写方法
继承dataset就是要实现/重写__init__  __getitem__
'''
# class MyData(Dataset):
#
#     def __init__(self):
#         pass
#
#     def __getitem__(self, item):
#         pass


class MyData(Dataset):

    def __init__(self, root_dir, label_dir):
        # self就是把root dir变成一个class中全部def都可以使用的全局变量
        # 设root_dir=dataset/train
        self.root_dir = root_dir
        # 设label_dir=ants
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        # if label dir = ants, then path=dataset/train/ants
        # os.listdir could list all ants img 的名字 如0013035.jpg
        # img_path是一个list
        self.img_path = os.listdir(self.path)

    def __getitem__(self, idx):
        """
        对MyData对象使用索引操作就会自动来到这个函数下边，双下划线是python中的魔法函数
        :param idx:
        :return:
        """
        img_name = self.img_path[idx]
        # 名字是0013035.jpg的图片的路径
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label

    # 再写一个获取数据集长度的魔法函数
    def __len__(self):
        return len(self.img_path)


'''
demo4:获取蚂蚁数据集dataset
'''
root_dir = "./dataset/train"
label_dir = "ants"
ants_dataset = MyData(root_dir, label_dir)
# print(ants_dataset[0])
# # output:(<PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=768x512 at 0x28C66665EE0>, 'ants')
# img, label = ants_dataset[0]
# img.show()
# print(label)

'''
demo5：再来获取蜜蜂的数据集
'''
root_dir = "./dataset/train"
label_dir = "bees"
bees_dataset = MyData(root_dir, label_dir)

'''
demo6:dataset数据集拼接
'''
train_dataset = ants_dataset + bees_dataset
