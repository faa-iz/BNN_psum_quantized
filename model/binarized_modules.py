import torch
import torch.nn as nn
import math
from torch.autograd import Variable
from torch.autograd import Function
#import torch.nn._functions as tnnf
import torch.nn.functional as F
import numpy as np
from .backward_ste import *
import model.targetprop as tp


def Quantize(tensor,quant_mode='det',  params=None, numBits=8):
    tensor.clamp_(-2**(numBits-1),2**(numBits-1))
    if quant_mode=='det':
        tensor=tensor.mul(2**(numBits-1)).round().div(2**(numBits-1))
    else:
        tensor=tensor.mul(2**(numBits-1)).round().add(torch.rand(tensor.size()).add(-0.5)).div(2**(numBits-1))
        quant_fixed(tensor, params)
    return tensor

def Binarize(tensor,quant_mode='det'):
    if quant_mode=='det':
        return tensor.sign()
    else:
        return tensor.add_(1).div_(2).add_(torch.rand(tensor.size()).add(-0.5)).clamp_(0,1).round().mul_(2).add_(-1)

class BinarizeFunc(torch.autograd.Function):
    '''
    Binarize the input activations and calculate the mean across channel dimension.
    '''

    @staticmethod
    def __init__(self, backward_ste, **kwargs):
        self.ste = BackwardSTE.get_backward_func(backward_ste)

        if (self.ste == BackwardSTE.swish_backward):
            self.across_filters = kwargs.get('across_filters', True)
            self.beta_init = kwargs.get('beta_init', 5)

    @staticmethod
    def forward(self, input):
        self.save_for_backward(input)
        if (self.ste == BackwardSTE.swish_backward):
            if not hasattr(self, 'beta'):
                self.input_dim = input.shape[1]
                if self.across_filters:
                    if (len(input.shape) == 4):
                        init = torch.ones(1, self.input_dim, 1, 1).cuda() * self.beta_init
                    elif (len(input.shape) == 2):
                        init = torch.ones(1, self.input_dim).cuda()
                    else:
                        init = torch.ones(1).cuda()
                else:
                    init = self.beta_init
                self.beta = init
        input = input.sign()
        return input

    @staticmethod
    def backward(self, grad_output):
        input, = self.saved_tensors
        if (self.ste == BackwardSTE.swish_backward):
            grad_input, self.beta = self.ste(input, grad_output, self.beta)
        else:
            grad_input = self.ste(input, grad_output)

        return grad_input

class SwishBinarize(nn.Module):
    def __init__(self, **kwargs):
        super(SwishBinarize, self).__init__()
        self.beta_init = kwargs.get('beta_init', 1.0)
        self.ss = SwishSign(**kwargs)

    def forward(self, x):
        if (self.training):
            return self.ss(x)
        else:
            return x.sign()


class BinActive(nn.Module):
    def __init__(self, backward_ste, **kwargs):
        super(BinActive, self).__init__()
        self.backward_ste = backward_ste
        self.kwargs = kwargs

    #@staticmethod
    def forward(self, x):
        return BinarizeFunc(self.backward_ste, **self.kwargs).apply(x)

    #@staticmethod
    def extra_repr(self):
        s = ('{backward_ste}, {}')
        t = ''
        for k, v in self.kwargs.items():
            t += str(k) + '=' + str(v)
            t += ', '
        t = t.strip(' ')
        return s.format(t, backward_ste=self.backward_ste)

class tanh(nn.Module):
    def __init__(self, beta_init):
        super(tanh, self).__init__()
        self.beta_init = beta_init
    def forward(self, x):
        return torch.tanh(self.beta_init * x)

class SwishSign(nn.Module):
    def __init__(self, beta_init=10, uniform=True, train_beta=True, inplace=False, **kwargs):
        super(SwishSign, self).__init__()
        self.inplace = True
        self.beta_init = beta_init
        self.uniform = uniform
        self.train_beta = train_beta
    def forward(self, x):
        if not hasattr(self, 'beta'):
            self.input_dim = x.shape[1]
            if not self.uniform:
                if (len(x.shape) == 4):
                    init = torch.ones(1, self.input_dim, 1, 1).cuda() * self.beta_init
                elif (len(x.shape) == 2):
                    init = torch.ones(1, self.input_dim).cuda() * self.beta_init
                else:
                    init = torch.ones(1).cuda() * self.beta_init
            else:
                init = torch.ones(1).cuda() * self.beta_init

            self.beta = nn.Parameter(init,
                        requires_grad=self.train_beta)

        y =  2 * F.sigmoid(self.beta * x) + 2 * x * self.beta * \
        (F.sigmoid(self.beta * x) * (1-F.sigmoid(self.beta * x))) - 1
        return y

    def extra_repr(self):
        inplace_str = ', inplace' if self.inplace else ''
        return 'lower={}, upper={}{}'.format(self.lower, self.upper, inplace_str)

def delete_elements_dict(vals, kwargs):
    for k in vals:
        if k in kwargs:
            del kwargs[k]

def keep_elements_dict(vals, kwargs):
    t = list(kwargs.keys())
    for k in t:
        if k not in vals:
            del kwargs[k]

class BinaryLinear(nn.Linear):

    def __init__(self, *kargs, **kwargs):
        self.scale = kwargs.get('scale', False)
        self.compute_scale = kwargs.get('compute_scale', 'topk')
        self.full_precision = kwargs.get('full_precision', True)
        keep_elements_dict(['stride', 'padding', 'kernel_size'], kwargs)
        super(BinaryLinear, self).__init__(*kargs, **kwargs)

    def forward(self, input):
        if not self.full_precision:
            input.data=Binarize(input.data)
            if self.scale and not hasattr(self, 'alpha'):
                if self.compute_scale == 'mean':
                    with torch.no_grad():
                        tk = torch.mean(self.weight.view(self.weight.shape[0], -1), dim=1)
                elif self.compute_scale == 'topk':
                    with torch.no_grad():
                        p = 0.5
                        k = torch.ceil((1-p) * torch.prod(torch.tensor(self.weight.shape[1:]), dtype=torch.float))
                        tk = torch.topk(torch.abs(self.weight.view(self.weight.shape[0], -1)), int(k), dim=1)[0][:, -1]
                elif self.compute_scale != "":
                    with torch.no_grad():
                        val = float(self.compute_scale)
                        tk = val * torch.ones(self.weight.shape[0], requires_grad=True)
                else:
                    raise NotImplementedError("compute_scale not implemented")
                tk = tk.to(self.weight.device)

                self.alpha = nn.Parameter(tk[:, None], requires_grad=True)


            if not hasattr(self.weight,'org'):
                self.weight.org=self.weight.data.clone()

            self.weight.data = Binarize(self.weight.org)
            weight = self.weight

            if (self.scale):
                weight = self.alpha * self.weight
        else:
            weight = self.weight

        out = nn.functional.linear(input, weight)
        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)

        return out

class BinaryConv2d(nn.Conv2d):
    def __init__(self, *kargs, **kwargs):

        self.scale = kwargs.get('scale', False)
        self.compute_scale = kwargs.get('compute_scale', 'topk')

        self.full_precision = kwargs.get('full_precision', True)
        self.is_ss = False
        self.activation = kwargs.get('activation')

        keep_elements_dict(['stride', 'padding', 'kernel_size', 'bias'], kwargs)

        super(BinaryConv2d, self).__init__(*kargs, **kwargs)

        if (self.activation == 'SS'):
            self.is_ss = True
            self.Binarize = SwishSign(**kwargs)
        else:
            self.Binarize = Binarize

    def forward(self, input):
        if not self.full_precision:
            if self.scale and not hasattr(self, 'alpha'):

                if self.compute_scale == "mean":
                    with torch.no_grad():
                        tk = torch.mean(self.weight.view(self.weight.shape[0], -1), dim=1)

                elif self.compute_scale == "topk":
                    with torch.no_grad():
                        p = 0.5
                        k = torch.ceil((1-p) * torch.prod(torch.tensor(self.weight.shape[1:]), dtype=torch.float))
                        tk = torch.topk(torch.abs(self.weight.view(self.weight.shape[0], -1)), int(k), dim=1)[0][:, -1]
                elif type(self.compute_scale) == float:
                    val = float(self.compute_scale)
                    tk = val * torch.ones(self.weight.shape[0])
                else:
                    raise NotImplementedError("compute_scale not implemented")
                tk = tk.to(self.weight.device)
                self.alpha = nn.Parameter(tk[:, None, None, None], requires_grad=True)
            if (self.is_ss):
                input = self.Binarize(input)
            else:
                input.data = self.Binarize(input.data)


            if not hasattr(self.weight,'org'):
                self.weight.org=self.weight.data.clone()

            if (self.is_ss):
                self.weight = self.Binarize(self.weight)
            else:
                self.weight.data = self.Binarize(self.weight.org)
            weight = self.weight

            if (self.scale):
                weight = self.alpha * self.weight
        else:
            weight = self.weight

        out = nn.functional.conv2d(input, weight, None, self.stride,
                                   self.padding, self.dilation, self.groups)
        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out
