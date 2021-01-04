'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import ir_1w1a


__all__ = ['resnet18', 'ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

def split_tensor_128(xp):
    x1 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2)
    x2 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2)
    x3 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2)
    x4 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2)
    x5 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2)
    x6 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2)
    x7 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2)
    x8 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2)
    x9 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2)

    return x1,x2,x3,x4,x5,x6,x7,x8,x9

def split_tesnsor_384(xp,max_size = 128):
    x1 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2).narrow(1,0,max_size)
    x2 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2).narrow(1,0,max_size)
    x3 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2).narrow(1,0,max_size)
    x4 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2).narrow(1,0,max_size)
    x5 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2).narrow(1,0,max_size)
    x6 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2).narrow(1,0,max_size)
    x7 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2).narrow(1,0,max_size)
    x8 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2).narrow(1,0,max_size)
    x9 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2).narrow(1,0,max_size)
    x12 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2).narrow(1,max_size,max_size)
    x22 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2).narrow(1,max_size,max_size)
    x32 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2).narrow(1,max_size,max_size)
    x42 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2).narrow(1,max_size,max_size)
    x52 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2).narrow(1,max_size,max_size)
    x62 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2).narrow(1,max_size,max_size)
    x72 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2).narrow(1,max_size,max_size)
    x82 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2).narrow(1,max_size,max_size)
    x92 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2).narrow(1,max_size,max_size)
    x13 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2).narrow(1,max_size*2,max_size)
    x23 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2).narrow(1,max_size*2,max_size)
    x33 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2).narrow(1,max_size*2,max_size)
    x43 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2).narrow(1,max_size*2,max_size)
    x53 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2).narrow(1,max_size*2,max_size)
    x63 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2).narrow(1,max_size*2,max_size)
    x73 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2).narrow(1,max_size*2,max_size)
    x83 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2).narrow(1,max_size*2,max_size)
    x93 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2).narrow(1,max_size*2,max_size)

    return x1, x2, x3, x4, x5, x6, x7, x8, x9, x12, x22, x32, x42, x52, x62, x72, x82, x92, x13, x23, x33, x43, x53, x63, x73, x83, x93

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()

        max_size = 128
        groups = math.ceil(inplanes / max_size)

        self.padding1 = nn.ZeroPad2d(1)
        input_dem = inplanes
        inplanes = min(input_dem, max_size)

        self.conv1 = Binaryconv1x1(inplanes, planes, stride)
        self.conv2 = Binaryconv1x1(inplanes, planes, stride)
        self.conv3 = Binaryconv1x1(inplanes, planes, stride)
        self.conv4 = Binaryconv1x1(inplanes, planes, stride)
        self.conv5 = Binaryconv1x1(inplanes, planes, stride)
        self.conv6 = Binaryconv1x1(inplanes, planes, stride)
        self.conv7 = Binaryconv1x1(inplanes, planes, stride)
        self.conv8 = Binaryconv1x1(inplanes, planes, stride)
        self.conv9 = Binaryconv1x1(inplanes, planes, stride)
        # '''
        if (input_dem > 128):
            self.conv12 = Binaryconv1x1(inplanes, planes, stride)
            self.conv22 = Binaryconv1x1(inplanes, planes, stride)
            self.conv32 = Binaryconv1x1(inplanes, planes, stride)
            self.conv42 = Binaryconv1x1(inplanes, planes, stride)
            self.conv52 = Binaryconv1x1(inplanes, planes, stride)
            self.conv62 = Binaryconv1x1(inplanes, planes, stride)
            self.conv72 = Binaryconv1x1(inplanes, planes, stride)
            self.conv82 = Binaryconv1x1(inplanes, planes, stride)
            self.conv92 = Binaryconv1x1(inplanes, planes, stride)

            self.conv13 = Binaryconv1x1(inplanes, planes, stride)
            self.conv23 = Binaryconv1x1(inplanes, planes, stride)
            self.conv33 = Binaryconv1x1(inplanes, planes, stride)
            self.conv43 = Binaryconv1x1(inplanes, planes, stride)
            self.conv53 = Binaryconv1x1(inplanes, planes, stride)
            self.conv63 = Binaryconv1x1(inplanes, planes, stride)
            self.conv73 = Binaryconv1x1(inplanes, planes, stride)
            self.conv83 = Binaryconv1x1(inplanes, planes, stride)
            self.conv93 = Binaryconv1x1(inplanes, planes, stride)
        '''
        self.conv1 = ir_1w1a.IRConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = ir_1w1a.IRConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        '''

    def forward(self, x):
        out = F.hardtanh(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.hardtanh(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        self.bn2 = nn.BatchNorm1d(512*block.expansion)
        self.softmax = nn.LogSoftmax()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.bn2(out)
        out = self.linear(out)
        out = self.softmax(out)
        return out 


def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])


def test():
    net = ResNet18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            print()
# test()
