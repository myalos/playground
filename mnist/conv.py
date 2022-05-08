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

# 这个版本没有预处理
mnist_train = datasets.MNIST(root = '../data', train = True, transform = transforms.ToTensor(), download = True)
mnist_test = datasets.MNIST(root = '../data', train = False, transform = transforms.ToTensor(), download = True)
torch.manual_seed(215)



# 根据resnet的结构来设计的
class ANGNet(nn.Module):
    def __init__(self):
        super(ANGNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size = 3, padding = 1, bias = False)

    def forward(self, x):
        pass


