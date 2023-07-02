import torch
import torchvision
from torch import nn

# 一般把神经网络单写一个包
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )
    def forward(self, input):
        output = self.seq(input)
        return output
