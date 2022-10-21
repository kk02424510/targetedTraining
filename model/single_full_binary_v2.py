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

def make_model():
    return Net()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.baseChannel = opt.ch
        self.input_Y = nn.Conv2d(
            in_channels=1, out_channels=self.baseChannel, kernel_size=3, stride=1, padding=1, bias=True)
        
        # self.relu = nn.ReLU6()
        

        self.b1_conv_1 = BinarizeConv2d(in_channels=self.baseChannel, out_channels=self.baseChannel,
                                  kernel_size=3, stride=1, padding=1, bias=True, dilation=1, groups=self.baseChannel)
        self.b1_conv_2 = BinarizeConv2d(in_channels=self.baseChannel, out_channels=self.baseChannel,
                                  kernel_size=3, stride=1, padding=1, bias=True, dilation=1, groups=self.baseChannel)
                                #   kernel_size=3, stride=1, padding=2, bias=False, dilation=2, groups=self.baseChannel)
        self.b1_conv_3 = BinarizeConv2d(in_channels=self.baseChannel, out_channels=self.baseChannel,
                                  kernel_size=3, stride=1, padding=1, bias=True, dilation=1, groups=self.baseChannel)
                                #   kernel_size=3, stride=1, padding=5, bias=False, dilation=5, groups=self.baseChannel)
        

        self.p_conv_1 = BinarizeConv2d(in_channels=self.baseChannel, out_channels=self.baseChannel,
                                  kernel_size=1, stride=1, padding=0, bias=True, dilation=1)
        self.p_conv_2 = BinarizeConv2d(in_channels=self.baseChannel, out_channels=self.baseChannel,
                                  kernel_size=1, stride=1, padding=0, bias=True, dilation=1)
        self.p_conv_3 = BinarizeConv2d(in_channels=self.baseChannel, out_channels=self.baseChannel,
                                  kernel_size=1, stride=1, padding=0, bias=True, dilation=1)
        
        self.bn0 = nn.BatchNorm2d(self.baseChannel) 
        self.bn1 = nn.BatchNorm2d(self.baseChannel) 
        self.bn2 = nn.BatchNorm2d(self.baseChannel) 
        self.bn3 = nn.BatchNorm2d(self.baseChannel) 
        self.bn4 = nn.BatchNorm2d(self.baseChannel) 
        self.bn5 = nn.BatchNorm2d(self.baseChannel) 
        self.bn6 = nn.BatchNorm2d(self.baseChannel) 
        
        self.tanh1 = nn.Hardtanh(inplace=True)
        self.tanh2 = nn.Hardtanh(inplace=True)
        self.tanh3 = nn.Hardtanh(inplace=True)
        self.tanh4 = nn.Hardtanh(inplace=True)
        # self.tanh5 = nn.Hardtanh(inplace=True)
        
        self.output_Y = nn.Conv2d(
            in_channels=self.baseChannel, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True)

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
        Y_out = self.bn0(Y_out)
        Y_out = self.tanh4(Y_out)

        yr = Y_out.clone()

        '''block1'''
        Y_out = self.b1_conv_1(Y_out)
        Y_out = self.bn1(Y_out)
        Y_out = self.tanh1(Y_out)
        Y_out = self.p_conv_1(Y_out)
        Y_out = self.bn2(Y_out)
        Y_out = self.tanh1(Y_out)

        Y_out = self.b1_conv_2(Y_out)
        Y_out = self.bn3(Y_out)
        Y_out = self.tanh1(Y_out)
        Y_out = self.p_conv_2(Y_out)
        Y_out = self.bn4(Y_out)
        Y_out = self.tanh2(Y_out)

        Y_out = self.b1_conv_3(Y_out)
        Y_out = self.bn5(Y_out)
        Y_out = self.tanh1(Y_out)
        Y_out = self.p_conv_3(Y_out)
        Y_out = self.bn6(Y_out)
        Y_out = self.tanh3(Y_out)
        

        Y_out = torch.add(Y_out, yr)

        Y_out = self.output_Y(Y_out)
        Y_out = torch.add(Y_out, residual_Y)

        return Y_out
