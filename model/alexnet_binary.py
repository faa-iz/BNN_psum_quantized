import torch
import torch.nn as nn
import torchvision.transforms as transforms
from .binarized_modules import *

__all__ = ['alexnet_binary_mod']


class partial_conv(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, kernel_size=11, stride=4, padding=2, full_precision=True,  **kwargs):
        super(partial_conv, self).__init__()

        if(in_planes==3):
            conv1 = BinaryConv2d(1, int(64 * self.ratioInfl), kernel_size=11, stride=4, padding=2, full_precision=True)
            conv2 = BinaryConv2d(1, int(64 * self.ratioInfl), kernel_size=11, stride=4, padding=2, full_precision=True)
            conv3 = BinaryConv2d(1, int(64 * self.ratioInfl), kernel_size=11, stride=4, padding=2, full_precision=True)
        else:
            if(kernel_size == 5):
                print("yet to code")

                if(in_planes>128):
                    print("yet to code")

                    if (in_planes > 256):
                        print("yet to code")


            elif (kernel_size == 3):
                print("yet to code")

                if (in_planes > 128):
                    print("yet to code")

                    if (in_planes > 256):
                        print("yet to code")

    def forward(self, x):
        output = []
        if(x.shape[1]==3):    #First layer split by channel
            x1 = x.narrow(1, 0, 1)
            x2 = x.narrow(1, 1, 1)
            x3 = x.narrow(1, 2, 1)
            output.append(self.conv1(x1))
            output.append(self.conv2(x2))
            output.append(self.conv3(x2))

        else:
            if(self.kernel_size == 5):
                print("yet to code")

                if (self.in_planes > 128):
                    print("yet to code")

                    if (self.in_planes > 256):
                        print("yet to code")
            elif (self.kernel_size == 3):
                print("yet to code")

                if (self.in_planes > 128):
                    print("yet to code")

                    if (self.in_planes > 256):
                        print("yet to code")

        out = merge(output)

        return out

class AlexNetOWT_BN(nn.Module):
    def init_model(self):
        model = self
        for m in model.modules():
            if isinstance(m, BinaryConv2d):
               nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                with torch.no_grad():
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
            elif isinstance(m, BinaryLinear):
                nn.init.xavier_normal_(m.weight)

    def __init__(self, num_classes=1000, **kwargs):
        super(AlexNetOWT_BN, self).__init__()
        self.ratioInfl=1
        nonlin = lambda : BinActive(backward_ste=kwargs.get('activation', 'STE'))
        self.features = nn.Sequential(
            BinaryConv2d(3, int(64*self.ratioInfl), kernel_size=11, stride=4, padding=2, full_precision=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(int(64*self.ratioInfl)),
            nonlin(),
            BinaryConv2d(int(64*self.ratioInfl), int(192*self.ratioInfl), kernel_size=5, padding=2, **kwargs),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(int(192*self.ratioInfl)),
            nonlin(),
            BinaryConv2d(int(192*self.ratioInfl), int(384*self.ratioInfl), kernel_size=3, padding=1, **kwargs),
            nn.BatchNorm2d(int(384*self.ratioInfl)),
            nonlin(),
            BinaryConv2d(int(384*self.ratioInfl), int(256*self.ratioInfl), kernel_size=3, padding=1, **kwargs),
            nn.BatchNorm2d(int(256*self.ratioInfl)),
            nonlin(),
            BinaryConv2d(int(256*self.ratioInfl), 256, kernel_size=3, padding=1, **kwargs),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(256),
            nonlin()

        )
        self.classifier = nn.Sequential(
            BinaryLinear(256 * 6 * 6, 4096, **kwargs),
            nn.BatchNorm1d(4096),
            nonlin(),
            BinaryLinear(4096, 4096, **kwargs),
            nn.BatchNorm1d(4096),
            nonlin(),
            BinaryLinear(4096, num_classes, full_precision=True),
            nn.BatchNorm1d(num_classes),
            nn.LogSoftmax()
        )

        #self.regime = {
        #    0: {'optimizer': 'SGD', 'lr': 1e-3,
        #        'weight_decay': 0, 'momentum': 0.9},
        #    10: {'lr': 1e-4},
        #    15: {'lr': 1e-4, 'weight_decay': 0},
        #    20: {'lr': 1e-5},
        #    25: {'lr': 1e-6}
        #}
        #self.regime = {
        #    0: {'optimizer': 'SGD', 'lr': 1e-2},
        #    30: {'lr': 1e-3},
        #    60: {'lr': 1e-4},
        #    90: {'lr': 1e-5},
        #    120: {'lr': 1e-6}
        #}
        self.regime = {
            0: {'optimizer': 'Adam', 'lr': 5e-3},
            20: {'lr': 1e-3},
            30: {'lr': 5e-4},
            35: {'lr': 1e-4},
            40: {'lr': 5e-5},
            45: {'lr': 1e-5}
        }

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.input_transform = {
            'train': transforms.Compose([
                transforms.Scale(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ]),
            'eval': transforms.Compose([
                transforms.Scale(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
            ])
        }
        self.init_model()

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 256 * 6 * 6)
        x = self.classifier(x)
        return x


def alexnet_binary_mod(**kwargs):
    return AlexNetOWT_BN(**kwargs)
