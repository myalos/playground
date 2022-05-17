# 这个里面是
# pytorch.org/tutorials/intermediate/tensorboard_tutorial.html

# 首先比较惊艳的地方就是torch.utils里面居然有tensorboard

import sys
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
from time import time
import argparse
from IPython import embed
import argparse
from tqdm import tqdm
import os
from torch.utils.tensorboard import SummaryWriter

# writer = SummaryWriter('runs')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))
    ])
transform1 = transforms.Compose([
    transforms.ToTensor(),
    ])

trainset = torchvision.datasets.FashionMNIST('../data', train = True, transform = transform)
trainset1 = torchvision.datasets.FashionMNIST('../data', train = True, transform = transform1)
testset = torchvision.datasets.FashionMNIST('../data', train = False, transform = transform)

train_dataloader = DataLoader(trainset, batch_size = 64, shuffle = True, num_workers = 0)
train_dataloader1 = DataLoader(trainset1, batch_size = 64, shuffle = False, num_workers = 0)
test_dataloader = DataLoader(testset, batch_size = 64, num_workers = 0)


#train_iter = iter(train_dataloader)
#images, labels = train_iter.next()
#train_iter1 = iter(train_dataloader1)
#images1, labels1 = train_iter1.next()
#
#img_grid = torchvision.utils.make_grid(images, nrow = 2)
#img_grid1 = torchvision.utils.make_grid(images1, nrow = 2)

# make_grid 受到Normalize的影响，nrow参数表示一行显示图片的张数，然后padding参数表示相邻图片的间隔像素
#writer.add_image('有Normalize', img_grid)
#writer.add_image('没有Normalize', img_grid1)


# add_graph 来查看model structure
# 来个最简单的LeNet
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

model = LeNet()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.003, momentum = 0.9)
# 这个add_graph 可以将节点展开，优点是可以看到节点的input和output的形状，缺点是看不到每个节点weight的形状，比如上面就看不到fc1的weight的形状
#writer.add_graph(model, images) # 这个graph 好像不需要给名字啊

# 接着是Projector 感觉这个挺有用的啊
def select_n_random(data, labels, n = 100):
    perm = torch.randperm(len(data))
    return data[perm][:n], labels[perm][:n]

#imgimg = trainset.data[:10].unsqueeze(1)
#gridgrid = torchvision.utils.make_grid(imgimg, nrow = 5)
# 用原始数据画图效果不错的，不知道这个和ToTensor后的是不是一样的
#writer.add_image('用原始数据来画图', gridgrid)

#images, labels = select_n_random(trainset.data, trainset.targets)
#class_labels = [classes[lab] for lab in labels]
# 这里是将28 * 28维度的数据进行投影
#features = images.view(-1, 28 * 28)
# 注意这里需要unsqueeze一下
# 效果很不错啊，里面可以用PCA 也可以用t-SNE，现在发现t-SNE是需要训练的, 里面提供的参数是Perplexity，Learning rate和Supervise 三种
#writer.add_embedding(features, metadata = class_labels, label_img = images.unsqueeze(1))

# 最后这两个非常的重要啊
# 下面是用tensorboard来tracking模型
# 这里一个是用add_scalar来代替print
# 用writer.add_figure来将matplotlib的Figure进行可视化

# 工具函数生成概率，实际中我没有用这个函数
def images_to_prob(net, images):
    output = net(images)
    # max函数 max(tensor, 0) 返回的是每列的最大值，返回值有两个第一个是值，第二个是索引，0改成1就变成每行的了，下面的preds_tensor就是索引了
    # 感觉这个和output.argmax(dim = 1) 没啥区别啊
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy())
    return preds, [F.softmax(el, dim = 0)[i].item() for i, el in zip(preds, output)]


# 这个真的不错啊
# 这个主要是生成预测的概率和标签，然后对照真实值label进行画图
def plot_classes_preds(net, images, labels):
    output = net(images)
    size = images.size(0)
    preds = output.argmax(dim = 1)
    probs = F.softmax(output, dim = 1)
    fig = plt.figure(figsize = (12, 12))
    plt.subplots_adjust(wspace = 0.7, hspace = 0.7)
    for i in range(size):
        ax = fig.add_subplot(8, 8, i + 1, xticks = [], yticks = [])
        npimg = (images[i].squeeze() / 2 + 0.5).numpy()
        ax.imshow(npimg, cmap = 'Greys')
        ax.set_title(f'{classes[preds[i]]}, {probs[i][preds[i]] * 100.0:.1f}%\n(label : {classes[labels[i]]})', color = ("green" if preds[i].item() == labels[i].item() else "red"))
    return fig


def train(net, dataloader, loss_fn, optimizer, epoch = None):
    net.train()
    running_loss = 0.0
    batch_num = len(dataloader)
    for data, label in dataloader:
        output = net(data)
        optimizer.zero_grad()
        loss = loss_fn(output, label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    if epoch:
        writer.add_scalar("training loss", running_loss / batch_num, epoch)

def test(net, dataloader, loss_fn, epoch = None):
    net.eval()
    batch_num = len(dataloader)
    total_size = len(dataloader.dataset)
    test_loss, correct, display = 0.0, 0, True
    with torch.no_grad():
        for data, label in dataloader:
            output = net(data)
            test_loss += loss_fn(output, label).item()
            correct += (output.argmax(dim = 1) == label).sum().item()
            if epoch and display:
                writer.add_figure("prediction vs actual", plot_classes_preds(model, data, label), global_step = epoch)
                display = False
    test_loss, correct = test_loss / batch_num, correct / total_size
    # 画的loss是以epoch为单位的，网页上的loss是以batch为单位的
    if epoch:
        writer.add_scalar("test loss", test_loss, epoch)
        writer.add_scalar("acc", correct, epoch)
    return correct


# 关于PR曲线的知识是 blog.csdn.net/b876144622/article/details/80009867
# 最后的准确率可达0.97（后来发现这个0.97用的是训练集的数据)
# tensorboard默认是只能画10张图，使用--samples_per_plugin=images=80 可以改最大值
if os.path.exists('model.pth'):
    _state = torch.load('model.pth')
    model.load_state_dict(_state)
    correct = test(model, test_dataloader, loss_fn, None)
    print(f'准确率为{correct:6f}')
else:
    for epoch in tqdm(range(1, 81)):
        train(model, train_dataloader, loss_fn, optimizer, epoch)
        test(model, test_dataloader, loss_fn, epoch)
    correct = test(model, test_dataloader, loss_fn, epoch = None)
    print(f'测试集上面的准确是{correct}') # 0.8927 但这个不是最高的 最高的有90.几
    # tensorboard上面看到的曲线test loss是一个U型 train loss一直下降，这个就是一个典型的过拟合的曲线
    # 30个epoch应该就差不多了
    torch.save(model.state_dict(), 'model.pth')


# 最后的部分是画PR曲线，PR曲线纵坐标是Precision 横坐标是Recall
# 需要知道预测的概率，预测标签和真实标签
# writer = SummaryWriter('runs')
def plot_pr_curve(net, dataloader):
    net.eval()
    # 需要两个列表
    class_labels, class_probs = [], []
    with torch.no_grad():
        for data, label in dataloader:
            output = net(data)
            class_probs.append(F.softmax(output.cpu(), dim = 1))
            class_labels.append(label.cpu())
    # 由test_probs就可以得到预测的值
    test_probs = torch.cat(class_probs)
    test_labels = torch.cat(class_labels)

    for class_index in range(10):
        # 四个参数 一个是str文字，一个是bool数组，一个是概率，一个是step
        # 这个pr曲线是一个单类的问题
        # 有了truth probs就可以算TP FP TN FN
        truth = test_labels == class_index
        probs = test_probs[:, class_index]
        pred_total = test_probs.argmax(dim = 1) == class_index

        TP = (pred_total * truth).sum().item()
        FP = (pred_total * ~truth).sum().item()
        TN = (~pred_total * ~truth).sum().item()
        FN = (~pred_total * truth).sum().item()
        print(f'对于类{classes[class_index]}')
        print(f'True Positive是：{TP}')
        print(f'False Positive是：{FP}')
        print(f'True Negative是：{TN}')
        print(f'False Negative是：{FN}')
        print(f'根据softmax precision是：{TP / (TP + FP)}')
        print(f'根据softmax recall是：{TP / (TP + FN)}')
        # 这个是将预测看成两类问题，要么是class_index 要么不是class_index
        # 完全忽略了其他类的预测概率 比如class_index的概率是0.3 我感觉其余概率都小于0.3和有一个概率是大于0.3的效果应该不一样
        # writer.add_pr_curve(classes[class_index], truth, probs, global_step = 0)
        # writer.add_pr_curve(classes[class_index], global_step = 0)

plot_pr_curve(model, test_dataloader)
# writer.close()

# 下面是在towardsdatascience.com/a-complete-guide-to-using-tensorboard-with-pytorch-53cb2301e8c3
# 主要是在for循环里面申请了 tb = SummaryWriter(comment = comment)
# 然后 tb.add_hparam里面装满超参数

