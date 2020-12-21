import torch

import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import torchvision.models as models

import sys
import math
__all__ = ['densenet']

class Bottleneck(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(Bottleneck, self).__init__()
        interChannels = 4*growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1,
                               bias=False)
        global activation
        if (activation == "relu"):
            self.act1 = F.relu
        elif (activation == "htan"):
            self.act1 = nn.functional.hardtanh

        self.bn2 = nn.BatchNorm2d(interChannels)
        if (activation == "relu"):
            self.act2 = F.relu
        elif (activation == "htan"):
            self.act2 = nn.functional.hardtanh
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.bn1(x)
        out = self.act1(out)
        out = self.conv1(out)
        # out = self.conv1(F.relu(self.bn1(x)))

        # out = self.conv2(F.relu(self.bn2(out)))
        out = self.bn2(out)
        out = self.act2(out)
        out = self.conv2(out)

        out = torch.cat((x, out), 1)
        return out

class SingleLayer(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)
        global activation
        if (activation == "relu"):
            self.act1 = F.relu
        elif (activation == "htan"):
            self.act1 = nn.functional.hardtanh
    def forward(self, x):
        # out = self.conv1(F.relu(self.bn1(x)))
        out = self.bn1(x)
        out = self.act1(out)
        out = self.conv1(out)
        out = torch.cat((x, out), 1)
        return out

class Transition(nn.Module):
    def __init__(self, nChannels, nOutChannels):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1,
                               bias=False)
        global activation
        if (activation == "relu"):
            self.act1 = F.relu
        elif (activation == "htan"):
            self.act1 = nn.functional.hardtanh


    def forward(self, x):
        # out = self.conv1(F.relu(self.bn1(x)))
        out = self.bn1(x)
        out = self.act1(out)
        out = self.conv1(out)

        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):
    def __init__(self, growthRate, depth, reduction, nClasses, bottleneck):
        super(DenseNet, self).__init__()
        global activation
        nDenseBlocks = (depth-4) // 3
        if bottleneck:
            nDenseBlocks //= 2

        nChannels = 2*growthRate
        self.conv1 = nn.Conv2d(3, nChannels, kernel_size=3, padding=1,
                               bias=False)
        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans1 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans2 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate

        self.bn1 = nn.BatchNorm2d(nChannels)
        if (activation == "relu"):
            self.act1 = F.relu
        elif (activation == "htan"):
            self.act1 = nn.functional.hardtanh
        self.fc = nn.Linear(nChannels, nClasses)
        self.init_model()
        self.regime = {
            0: {'optimizer': 'SGD', 'lr': 1.e-1, 'weight_decay': 1e-4, 'momentum':0.9},
            150: {'lr': 1.e-2},
            225: {'lr': 1.e-3},
        }

    def init_model(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck(nChannels, growthRate))
            else:
                layers.append(SingleLayer(nChannels, growthRate))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.dense3(out)
        out = self.bn1(out)
        out = self.act1(out)
        out = torch.squeeze(F.avg_pool2d(out, 8))
        #out = torch.squeeze(F.avg_pool2d(F.relu(self.bn1(out)), 8))
        out = F.log_softmax(self.fc(out))
        return out
activation = 'relu'
def densenet(**kwargs):
    depth = kwargs.get('depth', 100)
    growthRate = kwargs.get('growth_rate', 12)
    reduction = kwargs.get('reduction', 0.5)
    dataset = kwargs.get('dataset', 'cifar10')
    activation_tmp = kwargs.get('activation', 'relu')
    global activation
    activation = activation_tmp
    #import pdb; pdb.set_trace()

    if (dataset == 'imagenet'):
        nClasses = 1000
    elif (dataset == 'cifar10'):
        nClasses = 10
    elif (dataset == 'cifar100'):
        nClasses = 100

    return DenseNet(growthRate=growthRate, depth=depth, reduction=0.5,
                            bottleneck=True, nClasses=nClasses)
