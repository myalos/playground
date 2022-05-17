# author : myalos
# time : 2022 05 07
# environment:
#   macbook air 2020
#   python 3.7
#   pytorch 1.11.0
#
# description:
#   这个里面是用DANN版的MLP来进行MNIST的训练
#   这是一个需要debug的代码，loss没有怎么降，dev acc一直都是0.06，跟没训练是一样的，train loss 大概在3.几基本没降
#   后来发现我的classifier 是用的reverse_feature
#   而我的domain classifer 是用的feature

from tensorboardX import SummaryWriter
import sys
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

# 不是那么好训啊
parser = argparse.ArgumentParser('MLP parameters')
parser.add_argument('--lr', help = 'learning rate', default = 0.003)
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

writer = SummaryWriter('log')

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

# 参考代码
# https://github.com/fungtion/DANN_py3/blob/master/model.py
class DANN(nn.Module):
    def __init__(self):
        super(DANN, self).__init__()
        self.feature = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.BatchNorm1d(256), nn.ReLU(inplace = True), nn.Linear(256, 64), nn.BatchNorm1d(64), nn.ReLU(inplace = True))
        self.classifier = nn.Linear(64, 10)
        self.domain = nn.Sequential(nn.Linear(64, 64), nn.BatchNorm1d(64), nn.ReLU(inplace = True), nn.Linear(64, 2))

    # 这个output要output两个输出
    def forward(self, input_data, alpha):
        feature = self.feature(input_data)
        # 不要忘了apply
        #reverse_feature = ReverseLayerF(feature, alpha)
        #class_output = self.classifier(feature)
        #之前是这么写的，这么写就有问题 根本就没有用到RLF
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.classifier(feature)
        domain_output = self.domain(reverse_feature)
        return class_output, domain_output



model = DANN()
model.to(device)
loss_label = nn.CrossEntropyLoss()
loss_domain = nn.CrossEntropyLoss()
loss_domain.to(device)
loss_label.to(device)
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
            out, _ = net(X, 0.1)
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

# 当有两个数据迭代器的时候，使用iter来读数据
for epoch in tqdm(range(args.epoch)):
    # 统计每个epoch的统计量
    train_loss = AverageMeter()
    dev_loss = AverageMeter()
    dev_acc = AverageMeter()
    len_dataloader = min(len(mnist_train_iter), len(svhn_train_iter))
    data_source_iter = iter(mnist_train_iter)
    data_target_iter = iter(svhn_train_iter)
    model.train()
    for i in range(len_dataloader):
        # 不知道这个alpha是怎么设计出来的
        p = float(i + epoch * len_dataloader) / args.epoch / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        s_data, s_label  = data_source_iter.next()
        s_data, s_label = s_data.to(device), s_label.to(device)
        batch_size = len(s_label)
        optim.zero_grad()
        domain_label = torch.zeros(batch_size).long()
        domain_label = domain_label.to(device)
        class_output, domain_output = model(s_data, alpha)
        err_s_label = loss_label(class_output, s_label)
        err_s_domain = loss_domain(class_output, domain_label)

        t_data, _ = data_target_iter.next()
        t_data = t_data.to(device)
        batch_size = len(t_data)
        domain_label = torch.ones(batch_size).long()
        domain_label = domain_label.to(device)
        _, domain_output = model(t_data, alpha)
        err_t_domain = loss_domain(domain_output, domain_label)
        err = err_t_domain + err_s_label + err_s_domain
        err.backward()
        optim.step()
        # 检查一下梯度
        train_loss.update(err.item(), batch_size)
    writer.add_histogram('last feature weight', model.feature[4].weight, epoch)
    writer.add_histogram('last feature grad', model.feature[4].weight.grad, epoch)
    writer.add_histogram('classifier weight', model.classifier.weight, epoch)
    writer.add_histogram('classifier grad', model.classifier.weight.grad, epoch)
    writer.add_histogram('domain weight', model.domain[0].weight, epoch)
    writer.add_histogram('domain grad', model.domain[0].weight.grad, epoch)
    train_loss_history.append(train_loss.avg())


    model.eval()
    with torch.no_grad():
        for X, y in svhn_test_iter:
            X, y = X.to(device), y.to(device)
            out, _ = model(X, 0.1) # 这个0.1是个dummy的
            _loss_ = loss_label(out, y)
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
plt.plot(range(1, 1 + len(dev_acc_history)), dev_acc_history, label = 'dev')
plt.legend()
plt.show()

# 训练之后MNIST 准确率是 0.2842
# 训练之后的SVHN的准确率是 0.0683
# dev loss是在涨的
# 后来发现apply没有加上去
# 重新训
