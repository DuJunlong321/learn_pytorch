import torch
import torchvision

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from module import *

# 上设备，选GPU还是CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 数据集
train_dataset = torchvision.datasets.CIFAR10("../dataset", train=True, transform=torchvision.transforms.ToTensor(), download=True)
train_dataloader = DataLoader(train_dataset, batch_size=64)
train_lenth = len(train_dataset)

test_dataset = torchvision.datasets.CIFAR10("../dataset",train=False, transform=torchvision.transforms.ToTensor(), download=True)
test_dataloader = DataLoader(test_dataset, batch_size=64)
test_lenth = len(test_dataset)

# 神经网络
tudui = Tudui()
tudui = tudui.to(device)

# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

# 优化器
learningrate = 0.01;
opt = torch.optim.SGD(tudui.parameters(),lr=learningrate)

# 一些参数
epochs = 1  #训练次数
total_train_step = 0 #训练步数
total_test_step = 0  #册数步数

# 可视化
writer = SummaryWriter("./logs")

# 训练
for epoch in range(epochs):

    # 开始训练
    tudui.train()
    print("----第{}次训练开始----".format(epoch))
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        output = tudui(imgs)
        loss = loss_fn(output, targets)

        # 优化器优化模型
        opt.zero_grad()
        loss.backward()
        opt.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            print("训练次数{},损失{}".format(total_train_step, loss))
            writer.add_scalar("train",loss, total_train_step)

    # 开始测试
    tudui.eval()
    total_test_loss = 0
    total_accuracy = 0
    # 在测试步骤,不进行梯度计算
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            output = tudui(imgs)
            loss = loss_fn(output, targets)
            total_test_loss += loss.item()
            accuracy = (output.argmax(1) == targets).sum()
            total_accuracy += accuracy
    print("整体测试集上的loss:{}".format(total_test_loss))
    print("整体测试集上的正确率accuracy:{}".format(total_accuracy/test_lenth))
    writer.add_scalar("test",total_test_loss,total_test_step)
    writer.add_scalar("accuracy", total_accuracy/test_lenth, total_test_step)
    total_test_step += 1

    # 保存模型
    torch.save(tudui, "tudui_{}.pth".format(epoch))
    print("模型已保存")


writer.close()

