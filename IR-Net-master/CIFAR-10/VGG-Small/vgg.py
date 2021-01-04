'''
VGG for CIFAR10. FC layers are removed.
(c) YANG, Wei
'''
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
from modules import ir_1w1a


__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19', 'vgg_small', 'vgg_small_1w1a'
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}


quantize = False
binarize = False





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

def split_tensor_256(xp,max_size = 128):
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


    return x1, x2, x3, x4, x5, x6, x7, x8, x9, x12, x22, x32, x42, x52, x62, x72, x82, x92

def split_tensor_384(xp,max_size = 128):
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


def split_tensor_512(xp,max_size = 128):
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
    x14 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2).narrow(1, max_size * 3, max_size)
    x24 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2).narrow(1, max_size * 3, max_size)
    x34 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2).narrow(1, max_size * 3, max_size)
    x44 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2).narrow(1, max_size * 3, max_size)
    x54 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2).narrow(1, max_size * 3, max_size)
    x64 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2).narrow(1, max_size * 3, max_size)
    x74 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2).narrow(1, max_size * 3, max_size)
    x84 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2).narrow(1, max_size * 3, max_size)
    x94 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2).narrow(1, max_size * 3, max_size)

    return x1, x2, x3, x4, x5, x6, x7, x8, x9, x12, x22, x32, x42, x52, x62, x72, x82, x92, x13, x23, x33, x43, x53, x63, x73, x83, x93, x14, x24, x34, x44, x54, x64, x74, x84, x94



class split_conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(split_conv, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

        self.padding1 = nn.ZeroPad2d(1)

        self.conv1= ir_1w1a.IRConv2d(128, out_channels, kernel_size=1, padding=0, bias=False)
        self.conv2 = ir_1w1a.IRConv2d(128, out_channels, kernel_size=1, padding=0, bias=False)
        self.conv3 = ir_1w1a.IRConv2d(128, out_channels, kernel_size=1, padding=0, bias=False)
        self.conv4 = ir_1w1a.IRConv2d(128, out_channels, kernel_size=1, padding=0, bias=False)
        self.conv5 = ir_1w1a.IRConv2d(128, out_channels, kernel_size=1, padding=0, bias=False)
        self.conv6 = ir_1w1a.IRConv2d(128, out_channels, kernel_size=1, padding=0, bias=False)
        self.conv7 = ir_1w1a.IRConv2d(128, out_channels, kernel_size=1, padding=0, bias=False)
        self.conv8 = ir_1w1a.IRConv2d(128, out_channels, kernel_size=1, padding=0, bias=False)
        self.conv9 = ir_1w1a.IRConv2d(128, out_channels, kernel_size=1, padding=0, bias=False)

        if(in_channels>128):
            self.conv1_1 = ir_1w1a.IRConv2d(128, out_channels, kernel_size=1, padding=0, bias=False)
            self.conv2_1 = ir_1w1a.IRConv2d(128, out_channels, kernel_size=1, padding=0, bias=False)
            self.conv3_1 = ir_1w1a.IRConv2d(128, out_channels, kernel_size=1, padding=0, bias=False)
            self.conv4_1 = ir_1w1a.IRConv2d(128, out_channels, kernel_size=1, padding=0, bias=False)
            self.conv5_1 = ir_1w1a.IRConv2d(128, out_channels, kernel_size=1, padding=0, bias=False)
            self.conv6_1 = ir_1w1a.IRConv2d(128, out_channels, kernel_size=1, padding=0, bias=False)
            self.conv7_1 = ir_1w1a.IRConv2d(128, out_channels, kernel_size=1, padding=0, bias=False)
            self.conv8_1 = ir_1w1a.IRConv2d(128, out_channels, kernel_size=1, padding=0, bias=False)
            self.conv9_1 = ir_1w1a.IRConv2d(128, out_channels, kernel_size=1, padding=0, bias=False)

            if (in_channels>256):
                self.conv1_2 = ir_1w1a.IRConv2d(128, out_channels, kernel_size=1, padding=0, bias=False)
                self.conv2_2 = ir_1w1a.IRConv2d(128, out_channels, kernel_size=1, padding=0, bias=False)
                self.conv3_2 = ir_1w1a.IRConv2d(128, out_channels, kernel_size=1, padding=0, bias=False)
                self.conv4_2 = ir_1w1a.IRConv2d(128, out_channels, kernel_size=1, padding=0, bias=False)
                self.conv5_2 = ir_1w1a.IRConv2d(128, out_channels, kernel_size=1, padding=0, bias=False)
                self.conv6_2 = ir_1w1a.IRConv2d(128, out_channels, kernel_size=1, padding=0, bias=False)
                self.conv7_2 = ir_1w1a.IRConv2d(128, out_channels, kernel_size=1, padding=0, bias=False)
                self.conv8_2 = ir_1w1a.IRConv2d(128, out_channels, kernel_size=1, padding=0, bias=False)
                self.conv9_2 = ir_1w1a.IRConv2d(128, out_channels, kernel_size=1, padding=0, bias=False)

                if(in_channels>384):
                    self.conv1_3 = ir_1w1a.IRConv2d(128, out_channels, kernel_size=1, padding=0, bias=False)
                    self.conv2_3 = ir_1w1a.IRConv2d(128, out_channels, kernel_size=1, padding=0, bias=False)
                    self.conv3_3 = ir_1w1a.IRConv2d(128, out_channels, kernel_size=1, padding=0, bias=False)
                    self.conv4_3 = ir_1w1a.IRConv2d(128, out_channels, kernel_size=1, padding=0, bias=False)
                    self.conv5_3 = ir_1w1a.IRConv2d(128, out_channels, kernel_size=1, padding=0, bias=False)
                    self.conv6_3 = ir_1w1a.IRConv2d(128, out_channels, kernel_size=1, padding=0, bias=False)
                    self.conv7_3 = ir_1w1a.IRConv2d(128, out_channels, kernel_size=1, padding=0, bias=False)
                    self.conv8_3 = ir_1w1a.IRConv2d(128, out_channels, kernel_size=1, padding=0, bias=False)
                    self.conv9_3 = ir_1w1a.IRConv2d(128, out_channels, kernel_size=1, padding=0, bias=False)




    def forward(self, input):

        out = []
        input = self.padding1(input)

        if(self.in_channels == 128):
            x1,x2,x3,x4,x5,x6,x7,x8,x9 = split_tensor_128(input)
        elif (self.in_channels == 256):
            x1, x2, x3, x4, x5, x6, x7, x8, x9, x12, x22, x32, x42, x52, x62, x72, x82, x92 = split_tensor_256(input)
        elif (self.in_channels == 384):
            x1, x2, x3, x4, x5, x6, x7, x8, x9, x12, x22, x32, x42, x52, x62, x72, x82, x92, x13, x23, x33, x43, x53, x63, x73, x83, x93 = split_tensor_384(input)
        elif (self.in_channels == 512):
            x1, x2, x3, x4, x5, x6, x7, x8, x9, x12, x22, x32, x42, x52, x62, x72, x82, x92, x13, x23, x33, x43, x53, x63, x73, x83, x93, x14, x24, x34, x44, x54, x64, x74, x84, x94 = split_tensor_512(input)

        out.append(self.conv1(x1))
        out.append(self.conv2(x2))
        out.append(self.conv3(x3))
        out.append(self.conv4(x4))
        out.append(self.conv5(x5))
        out.append(self.conv6(x6))
        out.append(self.conv7(x7))
        out.append(self.conv8(x8))
        out.append(self.conv9(x9))

        if(self.in_channels>128):
            out.append(self.conv1_1(x12))
            out.append(self.conv2_1(x22))
            out.append(self.conv3_1(x32))
            out.append(self.conv4_1(x42))
            out.append(self.conv5_1(x52))
            out.append(self.conv6_1(x62))
            out.append(self.conv7_1(x72))
            out.append(self.conv8_1(x82))
            out.append(self.conv9_1(x92))

            if (self.in_channels > 256):
                out.append(self.conv1_2(x13))
                out.append(self.conv2_2(x23))
                out.append(self.conv3_2(x33))
                out.append(self.conv4_2(x43))
                out.append(self.conv5_2(x53))
                out.append(self.conv6_2(x63))
                out.append(self.conv7_2(x73))
                out.append(self.conv8_2(x83))
                out.append(self.conv9_2(x93))

                if (self.in_channels > 384):
                    out.append(self.conv1_3(x14))
                    out.append(self.conv2_3(x14))
                    out.append(self.conv3_3(x14))
                    out.append(self.conv4_3(x14))
                    out.append(self.conv5_3(x14))
                    out.append(self.conv6_3(x14))
                    out.append(self.conv7_3(x14))
                    out.append(self.conv8_3(x14))
                    out.append(self.conv9_3(x14))



        output = torch.zeros(out[0].shape).cuda()
        for out_tensor in out:
            # out_tensor = self.bn2(out_tensor,groups=groups)
            # out_tensor = self.tanh2(out_tensor)
            #if quantize:
                #out_tensor = custom_quantize(out_tensor, num_bit)
            #elif binarize:
                #out_tensor = scale * Binarize(out_tensor)

            output = output + out_tensor


        return output



class VGG(nn.Module):

    def __init__(self, features, num_classes=1000):
        super(VGG, self).__init__()
        self.features = features
        self.bn2 = nn.BatchNorm1d(512 * block.expansion)
        self.nonlinear2 = nn.Hardtanh(inplace=True)
        self.classifier = nn.Linear(512, num_classes)
        self.bn3 = nn.BatchNorm1d(num_classes)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.bn2(x)
        x = self.nonlinear2(x)
        x = self.classifier(x)
        x = self.bn3(x)
        x = self.logsoftmax(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.Hardtanh(inplace=True)]
            else:
                layers += [conv2d, nn.Hardtanh(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'F': [128, 128, 'M', 512, 512, 'M'],
}


def vgg11(**kwargs):
    """VGG 11-layer model (configuration "A")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['A']), **kwargs)
    return model


def vgg11_bn(**kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    model = VGG(make_layers(cfg['A'], batch_norm=True), **kwargs)
    return model


def vgg13(**kwargs):
    """VGG 13-layer model (configuration "B")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['B']), **kwargs)
    return model


def vgg13_bn(**kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    model = VGG(make_layers(cfg['B'], batch_norm=True), **kwargs)
    return model


def vgg16(**kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['D']), **kwargs)
    return model


def vgg16_bn(**kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    model = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
    return model


def vgg19(**kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['E']), **kwargs)
    return model


def vgg19_bn(**kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    model = VGG(make_layers(cfg['E'], batch_norm=True), **kwargs)
    return model



class VGG_SMALL_1W1A(nn.Module):

    def __init__(self, num_classes=10):
        super(VGG_SMALL_1W1A, self).__init__()
        self.conv0 = nn.Conv2d(3, 128, kernel_size=3, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(128)

        #self.conv1 = ir_1w1a.IRConv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.conv1 = split_conv(128, 128, kernel_size=3, padding=1, bias=False)

        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(128)
        # self.nonlinear = nn.ReLU(inplace=True)
        self.nonlinear = nn.Hardtanh(inplace=True)

        #self.conv2 = ir_1w1a.IRConv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.conv2 = split_conv(128, 256, kernel_size=3, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(256)

        self.conv3 = ir_1w1a.IRConv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.conv3 = split_conv(256, 256, kernel_size=3, padding=1, bias=False)

        self.bn3 = nn.BatchNorm2d(256)

        #self.conv4 = ir_1w1a.IRConv2d(256, 512, kernel_size=3, padding=1, bias=False)
        self.conv4 = split_conv(256, 512, kernel_size=3, padding=1, bias=False)

        self.bn4 = nn.BatchNorm2d(512)

        #self.conv5 = ir_1w1a.IRConv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.conv5 = split_conv(512, 512, kernel_size=3, padding=1, bias=False)

        self.bn5 = nn.BatchNorm2d(512)

        self.fc = nn.Linear(512*4*4, num_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, ir_1w1a.IRConv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.nonlinear(x)
        x = self.conv1(x)
        x = self.pooling(x)
        x = self.bn1(x)
        x = self.nonlinear(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.nonlinear(x)
        x = self.conv3(x)
        x = self.pooling(x)
        x = self.bn3(x)
        x = self.nonlinear(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.nonlinear(x)
        x = self.conv5(x)
        x = self.pooling(x)
        x = self.bn5(x)
        x = self.nonlinear(x)
        # x = self.pooling(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x



def vgg_small_1w1a(**kwargs):
    model = VGG_SMALL_1W1A(**kwargs)
    return model
