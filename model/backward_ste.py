# Modified code from https://github.com/afriesen/ftprop/blob/master/
from enum import Enum, unique
from functools import partial
import numpy as np
import torch

def soft_swish(x, beta):
    loss = beta * (2 - beta * x * torch.tanh(beta * x / 2)) \
            / (1 + (torch.exp(beta * x) + torch.exp(-beta * x)) / 2)

    return loss

def soft_hinge(z, t, xscale=1.0, yscale=1.0):
    loss = yscale * torch.tanh(-(z * t).float() * xscale) + 1
    return loss



# @unique
class BackwardSTE(Enum):

    STE = 0#'htan'
    SWISH = 1#'swish'
    BIREAL = 2#'bireal'
    FTPROP = 3#'ftprop'
    HTANH = 4 # htanh(3x)
    @staticmethod
    def wt_hinge_backward(step_input, grad_output, target, is01):
        if target is None:
            target = -torch.sign(grad_output)
        assert False
        return dhinge_dz(step_input, target, margin=1), None

    @staticmethod
    def bireal_backward(input, grad_output):
        go = grad_output.clone()
        grad_input = go * (2 + 2 * input) * torch.lt(input, 0).float() *  torch.ge(input, -1).float() \
                   + go * (2 - 2 * input) * torch.ge(input, 0).float() * torch.lt(input, 1).float()
        return grad_input

    @staticmethod
    def ste_backward(input, grad_output):
        grad_input = grad_output.clone()
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0

        return grad_input, None

    @staticmethod
    def htanh_backward(input, grad_output):
        grad_input = grad_output.clone()
        grad_input[input.ge(1/3)] = 0
        grad_input[input.le(-1/3)] = 0

        return 3.0 * grad_input, None


    @staticmethod
    def swish_backward(input, grad_output, beta):
        grad_input = grad_output.clone()
        z = soft_swish(input, beta)
        grad_input = grad_output.clone() * z
        return grad_input, beta

    @staticmethod
    def sste_backward(step_input, grad_output, target, is01, a=1):
        if is01:
            grad_input = grad_output * torch.ge(step_input, 0).float() * torch.le(step_input, a).float()
        else:
            grad_input = grad_output * torch.le(torch.abs(step_input), a).float()
        return grad_input, None

    @staticmethod
    def sigmoid_backward(step_input, grad_output, target, is01, xscale=2.0, yscale=1.0):
        assert not is01
        if target is None:
            target = torch.sign(-grad_output)
        z = sigmoid(step_input, target, xscale=xscale, yscale=1.0)
        grad_input = z * (1 - z) * xscale * yscale * -target / grad_output.size(0)
        return grad_input, None

    @staticmethod
    def tanh_backward(step_input, grad_output,target=None, xscale=3.0, yscale=1.0):
        # assert not is01
        if target is None:
            target = torch.sign(-grad_output)
        z = soft_hinge(step_input, target, xscale=xscale, yscale=1.0) - 1
        grad_input = (1 - z * z) * xscale * yscale * -target / grad_output.size(0)
        return grad_input, None

    @staticmethod
    def get_backward_func(backward_ste):
        try:
            backward_ste = BackwardSTE[backward_ste]
        except:
            raise ValueError('specified targetprop rule ({}) has no backward function {}'.format(backward_ste, 'choose one of {}'.format([e.name for e in BackwardSTE])))

        if backward_ste == BackwardSTE.STE:
            ste_func = BackwardSTE.ste_backward
        elif backward_ste == BackwardSTE.BIREAL:
            ste_func = BackwardSTE.bireal_backward
        elif backward_ste == BackwardSTE.SWISH:
            ste_func = BackwardSTE.swish_backward
        elif backward_ste == BackwardSTE.FTPROP:
            ste_func = BackwardSTE.tanh_backward
        elif backward_ste == BackwardSTE.HTANH:
            ste_func = BackwardSTE.htanh_backward
        return ste_func
