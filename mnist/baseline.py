# author : myalos
# time : 2022 04 04

# environment:
#   macbook air 2020
#   python 3.7
#   pytorch 1.11.0

# description:
#   最基本的训练过程
#   MNIST的baseline 用的是线性层

import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from time import time

# 加载数据
mnist_train = datasets.MNIST(root = '../data', train = True, transform = transforms.ToTensor(), download = True)
mnist_test = datasets.MNIST(root = '../data', train = False, transform = transforms.ToTensor(), download = True)

# 设置可复现的seed
torch.manual_seed(215)

# 显示图片
def display(images, labels, row, column):
    figsize = (row * 2, column * 2)
    fig, axes = plt.subplots(row, column, figsize = figsize)
    axes = axes.flatten() # axes的类型是ndarray
    for i, (ax, image, label) in enumerate(zip(axes, images, labels)):
        ax.imshow(image.numpy().transpose([1, 2, 0]), cmap = 'gray')
        ax.set_title(label.item())
        # 将图像x轴y轴的图片给去掉
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
    plt.show()

display_size = 20
#print(type(mnist_train[0][0])) # Tensor
#print(type(mnist_train[0][1])) # int
images = torch.stack([mnist_train[x][0] for x in range(display_size)])
labels = torch.tensor([mnist_train[x][1] for x in range(display_size)])

# display(images, labels, 4, 5)

# 超参数设置
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

learning_rate = 0.03
total_epoch = 20
bs = 512

# 工具函数
def evaluate(net, test_iter):
    total_instance, correct = 0.0, 0
    for X, y in tqdm(test_iter):
        X, y = X.to(device), y.to(device)
        out = net(X)
        total_instance += X.shape[0]
        correct += (out.argmax(dim = 1) == y).sum().item()
    return correct / total_instance

train_iter = torch.utils.data.DataLoader(mnist_train, batch_size = bs, num_workers = 0, shuffle = True)
test_iter = torch.utils.data.DataLoader(mnist_test, batch_size = bs, num_workers = 0, shuffle = False)

model = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
loss = nn.CrossEntropyLoss()
model.to(device)
loss.to(device)
optim = torch.optim.SGD(model.parameters(), lr = learning_rate)


# 未训练时 准确率
print("未训练时的准确率", evaluate(model, test_iter)) #0.0445

input("准备开始训练！")

# 进行训练
start_time = time()
for epoch in tqdm(range(total_epoch)):
    for X, y in train_iter:
        X, y = X.to(device), y.to(device)
        optim.zero_grad()
        out = model(X)
        _loss_ = loss(out, y)
        _loss_.backward()
        optim.step()

end_time = time()

print(f'训练时间： {end_time - start_time}') # 91s

print(f'训练后的准确率：{evaluate(model, test_iter)}') # 0.9053
