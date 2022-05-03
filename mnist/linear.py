# author : myalos
# time : 2022 05 03

# environment:
#   macbook air 2020
#   python 3.7
#   pytorch 1.11.0

# description:
#   最基本的训练过程
#   这个是在baseline.py的基础上增加了参数的调整
#   在baseline.py基础上加上了loss function的画图显示

import sys
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from time import time
import argparse
from IPython import embed

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', default = 20, help="epoch of learning")
parser.add_argument('--lr', default=0.03, help="learning rate of learning")
parser.add_argument('--bs', default=512, help="batch size of learning")
parser.add_argument('--seed', type=int, default=215, help='random seed of learning')
parser.add_argument('--viz', help='visualize the data', action = "store_true")
parser = parser.parse_args()

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

display_size = 20
#print(type(mnist_train[0][0])) # Tensor
#print(type(mnist_train[0][1])) # int
images = torch.stack([mnist_train[x][0] for x in range(display_size)])
labels = torch.tensor([mnist_train[x][1] for x in range(display_size)])

if parser.viz:
    display(images, labels, 4, 5)

# 使用matplotlib 画出饼状图
# @2022-05-02
def drawPie(labels):
    from collections import Counter
    fig = plt.figure(2)
    data = Counter(labels)
    plt.pie(data.values(), labels = data.keys(), autopct = '%0.2f%%')


# 下面是调用drawPie 画出饼状图
if parser.viz:
    drawPie([mnist_train[x][1] for x in range(len(mnist_train))])

# 超参数设置
device = "cuda" if torch.cuda.is_available() else "cpu"
print("使用的设备是：", device)

learning_rate = parser.lr
total_epoch = parser.epoch
bs = parser.bs

# 工具函数
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
    return np.sum(pred == label) / len(pred)

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

# cite by https://www.github.com/pytorch/examples/blob/main/imagenet/main.py @ 375
from enum import Enum
class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    def __init__(self, name, fmt:':f', summary_type = Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' +  self.fmt +'} {avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        # 感觉这个写法好帅啊
        # 经测试这个self.__dict__只有成员变量，没有一些稀奇古怪的东西
        return fmtstr.format(**self.__dict__)


train_loss_history, eval_loss_history, train_acc_history, eval_acc_history = [], [], [], []
# 进行训练
start_time = time()
for epoch in tqdm(range(total_epoch)):
    train_loss = AverageMeter('Train Loss', ":.4e", Summary.NONE)
    eval_loss = AverageMeter('Eval Loss', ":.4e", Summary.NONE)
    train_acc = AverageMeter('Train Acc', ':6.2f', Summary.AVERAGE)
    eval_acc = AverageMeter('Test Acc', ':6.2f', Summary.AVERAGE)

    model.train()
    for X, y in train_iter:
        X, y = X.to(device), y.to(device)
        optim.zero_grad()
        out = model(X)
        _loss_ = loss(out, y)
        _loss_.backward()
        optim.step()
        train_loss.update(_loss_.item())
        train_acc.update((out.argmax(dim = 1) == y).sum().item() / X.shape[0], X.shape[0])
    train_loss_history.append(train_loss.avg)
    train_acc_history.append(train_acc.avg)
    model.eval()
    with torch.no_grad():
        correct = 0.0
        for X, y in test_iter:
            X, y = X.to(device), y.to(device)
            out = model(X)
            _loss_ = loss(out, y)
            eval_loss.update(_loss_.item(), X.shape[0])
            eval_acc.update((out.argmax(dim = 1) == y).sum().item() / X.shape[0], X.shape[0])
    eval_loss_history.append(eval_loss.avg)
    eval_acc_history.append(eval_acc.avg)

end_time = time()
print(f'训练时间： {end_time - start_time}') # 91s
print(f'训练后的准确率：{evaluate(model, test_iter, False)}') # 0.9053
# 训练时间是 1分28秒
print(f'训练后的混淆矩阵是：\n {evaluate(model, test_iter, True)}')

# 画出曲线
fig = plt.figure(3)
plt.subplot(2, 1, 1)
plt.title("loss")
plt.plot(range(1, 1 + len(train_loss_history)), train_loss_history, label = 'train')
plt.plot(range(1, 1 + len(eval_loss_history)), eval_loss_history, label = 'eval')
plt.legend()
plt.subplot(2, 1, 2)
plt.title("accuracy")
plt.plot(range(1, 1 + len(train_acc_history)), train_acc_history, label = 'train')
plt.plot(range(1, 1 + len(eval_acc_history)), eval_acc_history, label = 'eval')
plt.legend() # 这个重要啊，不然就不会显示label了
# 所有的图 最后画
plt.show()

