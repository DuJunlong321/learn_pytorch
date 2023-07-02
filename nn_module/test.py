import PIL
import torch
import torchvision.transforms
from PIL import Image
from module import Tudui

# 读取图片
img_path = "./img/img_2.png"
img = Image.open(img_path)
print(img)

# 转换成三通道RGB
img.convert('RGB')
print(img)

# 修改PIL img的尺寸 + 转换成tensor
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),
                               torchvision.transforms.ToTensor()])
img = transform(img)
print(img.shape)

# 修改tensor img的尺寸
img = torch.reshape(img, (1, 3, 32, 32))
print(img.shape)

img = img.cuda()

module = torch.load("tudui_0.pth")
module.eval()
with torch.no_grad():
    output = module(img)
print(output)
print(output.argmax(1))


