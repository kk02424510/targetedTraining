# python main.py --model Multi_04 --colortype yuv --loss 1*MSE --clip 0.4 --lr 0.1 --lr_decay 40 --gamma 0.1 --batch_size 64 --patch_size 80 --test_every 2500 --print_every 250 --epochs 160 --optimizer SGD --weight_decay 1e-4 --save_result --rgb_range 255
# python main.py --model Multi_03-ev2 --colortype yuv --loss 1*L1char --clip 0.4 --lr 0.01 --lr_decay 40 --gamma 0.1 --batch_size 32 --patch_size 80 --test_every1250 --print_every 125 --epochs 120 --optimizer SGD --weight_decay 1e-4 --save_result --multi --qp 10 --data_train DIV2K --n_train 900 --save qp10
# Base CNN Model + Recursive 3 times + residual block + dilated
import sys
import torch
import torch.nn as nn
from math import sqrt
from model import common
from option import opt
from .binarized_modules import BinarizeLinear,BinarizeConv2d
# from switchable_norm import SwitchNorm2d

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups
    
    # reshape
    x = x.view(batchsize, groups, 
        channels_per_group, height, width)

    # transpose
    # - contiguous() required if transpose() is used before view().
    #   See https://github.com/pytorch/pytorch/issues/764
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x

def Binaryconv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    # return BinarizeConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
    #                  padding=1, bias=False)
    return BinarizeConv2d(in_planes, out_planes, in_channels=self.baseChannel, out_channels=self.baseChannel,
                          kernel_size=3, stride=1, padding=1, bias=False, dilation=1, groups=self.baseChannel)

def make_model():
    return Net()
    

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # self.group = opt.groups
        self.baseChannel = opt.ch
        self.input_Y = nn.Sequential(
            nn.Conv2d(
            in_channels=1, out_channels=self.baseChannel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(self.baseChannel),
            nn.Hardtanh(inplace=True)
        )

        self.conv1 = nn.Sequential(
            BinarizeConv2d(
            in_channels=self.baseChannel, out_channels=self.baseChannel, kernel_size=3, stride=1, padding=1, bias=True, groups=self.baseChannel),
            nn.BatchNorm2d(self.baseChannel),
            nn.Hardtanh(inplace=True)
        )

        self.conv2 = nn.Sequential(
            BinarizeConv2d(
            in_channels=self.baseChannel, out_channels=self.baseChannel, kernel_size=3, stride=1, padding=1, bias=True, groups=self.baseChannel),
            nn.BatchNorm2d(self.baseChannel),
            nn.Hardtanh(inplace=True)
        )

        self.conv3 = nn.Sequential(
            BinarizeConv2d(
            in_channels=self.baseChannel, out_channels=self.baseChannel, kernel_size=3, stride=1, padding=1, bias=True, groups=self.baseChannel),
            nn.BatchNorm2d(self.baseChannel),
            nn.Hardtanh(inplace=True)
        )

        self.output_Y = nn.Sequential(
            nn.Conv2d(
            in_channels=self.baseChannel*2,out_channels=1, kernel_size=3, stride=1, padding=1, bias=True)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
        # ### If load retrined model
        if opt.pre_train != '.':
            pt = torch.load(opt.pre_train)
            for p in pt.keys():
                self.state_dict()[p].copy_(pt[p])
        # ###

    def forward(self, x):
        Y_out = x[:, 0, :, :].unsqueeze(1)
        residual_Y = Y_out.clone()
        Y_out = self.input_Y(Y_out)
        yr = Y_out.clone()
        Y_out = self.conv1(Y_out)
        Y_out = self.conv2(Y_out)
        Y_out = self.conv3(Y_out)
        # Y_out = torch.add(Y_out,yr)
        Y_out = torch.cat((Y_out,yr),1)
        Y_out = self.output_Y(Y_out)
        Y_out = torch.add(Y_out, residual_Y)
        
        return Y_out