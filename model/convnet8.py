# architecture replicated from DoReFaNet code at
# https://github.com/ppwwyyxx/tensorpack/blob/master/examples/DoReFa-Net/svhn-digit-dorefa.py
from collections import OrderedDict
import torch
import torch.nn as nn
from .binarized_modules import *

from torch.nn import Module

class ReshapeBatch(Module):
    """A simple layer that reshapes its input and outputs the reshaped tensor """
    def __init__(self, *args):
        super(ReshapeBatch, self).__init__()
        self.args = args

    def forward(self, x):
        return x.view(x.size(0), *self.args)

    def __repr__(self):
        s = '{}({})'
        return s.format(self.__class__.__name__, self.args)
__all__ = ['convnet8']
class ConvNet8(nn.Module):
    def __init__(self, num_classes=10, activation="ste", relu_first_layer=True, swish_sign=False, scale=False, compute_scale=False, full_precision=False, beta_init=10, uniform=True, train_beta=False, reg_type='l1', input_shape=(3, 32, 32)):
        super(ConvNet8, self).__init__()
        use_bn = True
        self.use_bn = use_bn
        bias = not use_bn

        if input_shape[1] == 40:
            pad0 = 0
            ks6 = 5
        elif input_shape[1] == 32:
            pad0 = 2
            ks6 = 4
        else:
            raise NotImplementedError('no other input sizes are currently supported')

        nonlin = lambda : BinActive(backward_ste=kwargs.get('activation', 'STE'))


        block0 = OrderedDict([
            ('conv0', nn.Conv2d(3, 48, kernel_size=5, padding=pad0)),  # padding = valid
            ('maxpool0', nn.MaxPool2d(2)),  # padding = same
            ('nonlin1', nn.ReLU(inplace=True) if relu_first_layer else nonlin())  # 18
        ])

        block1 = OrderedDict([
            ('batchnorm1', nn.BatchNorm2d(48, eps=1e-4)),
            ('nonlin1', nonlin()),
            ('conv1', BinaryConv2d(48, 64, kernel_size=3, padding=1, **kwargs)),  # padding = same
        ])

        block2 = OrderedDict([
            ('batchnorm2', nn.BatchNorm2d(64, eps=1e-4)),
            ('nonlin2', nonlin()),  # 9
            ('conv2', BinaryConv2d(64, 64, kernel_size=3, padding=1, **kwargs)),  # padding = same
            ('maxpool1', nn.MaxPool2d(2)),      # padding = same
        ])

        block3 = OrderedDict([
            ('batchnorm3', nn.BatchNorm2d(64, eps=1e-4)),
            ('nonlin3', nonlin()),  # 7
            ('conv3', BinaryConv2d(64, 128, kernel_size=3, padding=0, **kwargs)),  # padding = valid
        ])

        block4 = OrderedDict([
            ('batchnorm4', nn.BatchNorm2d(128, eps=1e-4)),
            ('nonlin4', nonlin()),
            ('conv4', BinaryConv2d(128, 128, kernel_size=3, padding=1, **kwargs)),  # padding = same
        ])

        block5 = OrderedDict([
            ('batchnorm5', nn.BatchNorm2d(128, eps=1e-4)),
            ('nonlin5', nonlin()),  # 5
            ('conv5', BinaryConv2d(128, 128, kernel_size=3, padding=0, **kwargs)),  # padding = valid

        ])

        block6 = OrderedDict([
            # ('dropout', nn.Dropout2d()),
            ('batchnorm6', nn.BatchNorm2d(128, eps=1e-4)),
            ('nonlin6', nn.ReLU(inplace=True) if relu_first_layer else nonlin()),
            ('conv6', nn.Conv2d( 128, 512, kernel_size=ks6, padding=0)),  # padding = valid
        ])

        block7 = OrderedDict([
            ('reshape_fc1', ReshapeBatch(-1)),
            ('fc1', nn.Linear(512, 10, bias=True)),
            ('softmax', nn.LogSoftmax())
        ])

        if not self.use_bn:
            del block1['batchnorm1']
            del block2['batchnorm2']
            del block3['batchnorm3']
            del block4['batchnorm4']
            del block5['batchnorm5']
            del block6['batchnorm6']

        self.all_modules = nn.Sequential(OrderedDict([
            ('block0', nn.Sequential(block0)),
            ('block1', nn.Sequential(block1)),
            ('block2', nn.Sequential(block2)),
            ('block3', nn.Sequential(block3)),
            ('block4', nn.Sequential(block4)),
            ('block5', nn.Sequential(block5)),
            ('block6', nn.Sequential(block6)),
            ('block7', nn.Sequential(block7)),
        ]))

        self.regime = {
            0: {'optimizer': 'Adam', 'lr':1e-3},
            50: {'lr':0.5e-3},
            100: {'lr': 1e-4},
            150: {'lr': 1e-5},
            280: {'lr': 1e-6},
        }

    def forward(self, x):
        x = self.all_modules(x)
        return x

def convnet8(**kwargs):
    num_classes, depth, dataset, scale, swish_sign, full_precision, activation, uniform, train_beta, beta_init, compute_scale, reg_type, relu_first_layer = map(kwargs.get, ['num_classes', 'depth', 'dataset', 'scale', 'swish_sign', 'full_precision', 'activation', 'uniform', 'train_beta', 'beta_init', 'compute_scale', 'reg_type', 'relu_first_layer'])
    if dataset == 'imagenet':
        num_classes = 1000
    elif dataset == 'cifar10':
        num_classes = 10
    return ConvNet8(num_classes, activation, relu_first_layer, swish_sign, scale, compute_scale, full_precision, beta_init, uniform, train_beta, reg_type)
