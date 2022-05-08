# author : myalos
# time : 2022 05 07
# environment:
#   macbook air 2020
#   python 3.7
#   pytorch 1.11.0
#
# description:
#   这个里面是用MLP来进行MNIST的训练
#
# code structure:
#   命令行解析
#   数据加载（灰度化 tensor化 归一化）
#   定义工具类函数 evaluate 饼状图 图像预览 训练loss记录部分
#   定义模型 损失函数 优化器
#   进行训练

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from time import time
import argparse
from IPython import embed
import argparse
from tqdm import tqdm
import os
from torch.autograd import Function

parser = argparse.ArgumentParser('MLP parameters')
parser.add_argument('--lr', help = 'learning rate', default = 0.03)
parser.add_argument('--bs', type = int, help = 'batch size', default = 128)
parser.add_argument('--epoch', type = int, help = 'training epoch', default = 30)
args = parser.parse_args()

# 实验可重复性设置
torch.manual_seed(42)

mnist_trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.1307, ), std = (0.3081, ))
    ])

svhn_trans = transforms.Compose([
        transforms.Resize(28),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.44552204, ), std = (0.19425017, ))
    ])

mnist_train = datasets.MNIST(root = '../data', train = True, transform = mnist_trans)
mnist_test = datasets.MNIST(root = '../data', train = False, transform = mnist_trans)


mnist_train_iter = DataLoader(mnist_train, shuffle = True, batch_size = 128, num_workers = 0)
mnist_test_iter = DataLoader(mnist_test, shuffle = False, batch_size = 128, num_workers = 0)

# 同样的方法来计算svhn_train的mean 和 std
# 看了文档后面发现 svhn数据集是没有train这个参数的
# 取而代之的是split这个参数 train test extra 三选一
svhn_train = datasets.SVHN(root = '../data', split = 'train', transform = svhn_trans)
svhn_test = datasets.SVHN(root = '../data', split = 'test', transform = svhn_trans)

svhn_train_iter = DataLoader(svhn_train, shuffle = True, batch_size = 128, num_workers = 0)
svhn_test_iter = DataLoader(svhn_test, shuffle = False, batch_size = 128, num_workers = 0)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'使用的设备是 {device}')

# DANN
class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class DANN(nn.Module):
    def __init__(self):
        super(DANN, self).__init__()
        self.feature = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.BatchNorm1d(256), nn.ReLU(inplace = True), nn.Linear(256, 64), nn.BatchNorm1d(64), nn.ReLU(inplace = True))
        self.classifier = nn.Linear(64, 10)
        self.domain = nn.Sequential(nn.Linear(64, 64), nn.BatchNorm1d(64), nn.ReLU(inplace = True), nn.Linear(64, 2))

    # 这个output要output两个输出
    def forward(self, x):
        # 在另一个文件里面重写一下
        pass

# 最近学的keras里面 可以写个函数build_model来和sklearn进行random search，pytorch说不定也可以，最后可以试着写一下
model = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.BatchNorm1d(256), nn.ReLU(inplace = True), nn.Linear(256, 64), nn.BatchNorm1d(64), nn.ReLU(inplace = True), nn.Linear(64, 10))

model.to(device)
loss = nn.CrossEntropyLoss()
loss.to(device)
optim = torch.optim.SGD(model.parameters(), lr = args.lr)

# 画出svhn 标签的分布
def drawPie(labels):
    from collections import Counter
    fig = plt.figure()
    data = Counter(labels)
    plt.pie(data.values(), labels = data.keys(), autopct = '%0.2f%%')

drawPie([svhn_train[i][1] for i in range(len(svhn_train))])

# evaluate 相关的函数
# 三个参数 模型 测试集 返回值的形式
def evaluate(net, test_iter, confusion_matrx = False):
    net.eval()
    pred, label = [], []
    with torch.no_grad():
        for X, y in tqdm(test_iter):
            X, y = X.to(device), y.to(device)
            out = net(X)
            pred.append(out.argmax(dim = 1).cpu().numpy())
            label.append(y.cpu().numpy())
    pred = np.concatenate(pred, axis = 0)
    label = np.concatenate(label, axis = 0)
    if confusion_matrx:
        from sklearn.metrics import confusion_matrix
        return confusion_matrix(label, pred)
    return np.sum(label == pred) / label.shape[0]

# meter相关的函数
class AverageMeter:
    def __init__(self):
        self.sum = 0.0
        self.num = 0

    def update(self, val, n = 1):
        self.sum = self.sum + val * n
        self.num = self.num + n

    def avg(self):
        return self.sum / self.num
# 这个mean和std 是三通道的0.4514 和 0.1993
# 如果先进行灰度化 那么mean和std是0.4453和0.1970
# 后来发现 要把图片先resize到28x28，resize之后重新计算了一下mean和std 发现值为0.44552204 和 0.19425017
# 也可以通过下面的方法来掉库来算
# X = np.concatenate([svhn_train[i][0] for i in range(len(svhn_train))], axis = 0)
# X.shape
# X.mean(), X.std()

# 0.0974 和 0.107828
print(f'训练之前MNIST上的准确率是: {evaluate(model, mnist_test_iter)}')
print(f'训练之前SVHN上的准确率是: {evaluate(model, svhn_test_iter)}')

# 训练过程
input("开始训练！")

# 四个history train_loss dev_loss train_acc dev_acc
train_loss_history, dev_loss_history, train_acc_history, dev_acc_history = [], [], [], []

start_time = time()

for epoch in tqdm(range(args.epoch)):
    # 统计每个epoch的统计量
    train_loss = AverageMeter()
    dev_loss = AverageMeter()
    train_acc = AverageMeter()
    dev_acc = AverageMeter()

    model.train()
    for X, y in mnist_train_iter:
        X, y = X.to(device), y.to(device)
        optim.zero_grad()
        out = model(X)
        _loss_ = loss(out, y)
        _loss_.backward()
        optim.step()
        train_loss.update(_loss_.item(), X.shape[0])
        train_acc.update((out.argmax(dim = 1) == y).sum().item() / X.shape[0], X.shape[0])
    train_loss_history.append(train_loss.avg())
    train_acc_history.append(train_loss.avg())
    model.eval()
    with torch.no_grad():
        for X, y in mnist_test_iter:
            X, y = X.to(device), y.to(device)
            out = model(X)
            _loss_ = loss(out, y)
            dev_loss.update(_loss_.item(), X.shape[0])
            dev_acc.update((out.argmax(dim = 1) == y).sum().item() / X.shape[0], X.shape[0])
    dev_loss_history.append(dev_loss.avg())
    dev_acc_history.append(dev_acc.avg())

end_time = time()
print(f'训练时间是: {end_time - start_time}')
print('训练后MNIST准确率: ', evaluate(model, mnist_test_iter))
print('训练后SVHN准确率: ', evaluate(model, svhn_test_iter))

fig = plt.figure(2)
plt.subplot(2, 1, 1)
plt.title("loss")
plt.plot(range(1, 1 + len(train_loss_history)), train_loss_history, label = 'train')
plt.plot(range(1, 1 + len(dev_loss_history)), dev_loss_history, label = 'dev')
plt.legend()
plt.subplot(2, 1, 2)
plt.title("acc")
plt.plot(range(1, 1 + len(train_acc_history)), train_acc_history, label = 'train')
plt.plot(range(1, 1 + len(dev_acc_history)), dev_acc_history, label = 'dev')
plt.legend()
plt.show()

# 运行时间340秒
# 运行前 MNIST 准确率 0.1118
# 运行前 SVHN 准确率 0.0598
# 运行后 MNIST 准确率 0.9822
# 运行后 SVHN 准确率 0.155385
# 看loss曲线发现 大概10个epoch 就可以收敛了
