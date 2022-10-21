import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
from torch.optim import lr_scheduler
import numpy as np
import torch.nn.functional as F
from math import sqrt
###############################################################################
# Functions
###############################################################################



class swish(nn.Module):
    def __init__(self):
        super(swish, self).__init__()
    
    def forward(self, x):
        return x * F.sigmoid(x)

def weights_init_vdsr(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, sqrt(2. / n))
        # init.normal(m.weight.data, 0.0, sqrt(2. / n))
    elif classname.find('Linear') != -1:
        init.uniform(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv2d') != -1:
        print(classname)
        init.uniform(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.uniform(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    elif init_type == 'vdsr':
        net.apply(weights_init_vdsr)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif layer_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - opt.niter) / float(opt.niter_decay+1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

def print_network(net):
    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(net)
    print('Total number of parameters: %d' % params)


class ResBlock(nn.Module):
    def __inin__(self, dim, ksize, act=nn.ReLU):
        super(ResBlock, self).__init__()
        self.conv_block = build_conv_block(dim, ksize, act)

        def build_conv_block(self, dim, ksize, act, padding_type='zero', use_bias=False):
            conv_block = []
            p = 0
            if padding_type == 'zero':
                p = ksize // 2
            elif padding_type == 'reflect':
                conv_block.append(nn.ReflectionPad2d(ksize // 2))
            elif padding_type == 'replicate':
                conv_block.append(nn.ReplicationPad2d(ksize // 2))
            conv_block.append(nn.Conv2d(in_channels=dim, out_channels=dim,
                                    kernel_size=ksize,
                                    stride=1, padding=p, bias=use_bias))
            conv_block.append(nn.BatchNorm2d(dim))
            conv_block.append(act(inplace=True))
            return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

###############################################################################

def quanWeight(module, n):
    module.mean = torch.mean(module.weight.data)
    module.delta = (torch.max(module.weight.data) - torch.min(module.weight.data)) / (2**n - 1)
    module.rn = torch.clamp(torch.round((module.weight.data - module.mean)/module.delta), -2**(n-1), 2**(n-1))
    return module.rn * module.delta + module.mean

def quanParam(module):
    module.preQuanWeight = module.weight.data.clone()
    module.weight.data.copy_(quanWeight(module))


###############################################################################