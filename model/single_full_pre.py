# python main.py --model Multi_04 --colortype yuv --loss 1*MSE --clip 0.4 --lr 0.1 --lr_decay 40 --gamma 0.1 --batch_size 64 --patch_size 80 --test_every 2500 --print_every 250 --epochs 160 --optimizer SGD --weight_decay 1e-4 --save_result --rgb_range 255
# python main.py --model Multi_03-ev2 --colortype yuv --loss 1*L1char --clip 0.4 --lr 0.01 --lr_decay 40 --gamma 0.1 --batch_size 32 --patch_size 80 --test_every1250 --print_every 125 --epochs 120 --optimizer SGD --weight_decay 1e-4 --save_result --multi --qp 10 --data_train DIV2K --n_train 900 --save qp10
# Base CNN Model + Recursive 3 times + residual block + dilated
import sys
import torch
import torch.nn as nn
from math import sqrt
from model import common
from option import opt

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
            in_channels=1, out_channels=self.baseChannel, kernel_size=3, stride=1, padding=1, bias=False)
        self.act = nn.ReLU()
        self.InstanceNorm = nn.InstanceNorm2d(self.baseChannel) 
        
        self.b1_conv_1 = nn.Conv2d(in_channels=self.baseChannel, out_channels=self.baseChannel,
                                  kernel_size=3, stride=1, padding=1, bias=False, dilation=1)
        self.b1_conv_2 = nn.Conv2d(in_channels=self.baseChannel, out_channels=self.baseChannel,
                                  kernel_size=3, stride=1, padding=1, bias=False, dilation=1)
                                #   kernel_size=3, stride=1, padding=2, bias=False, dilation=2, groups=self.baseChannel)
        self.b1_conv_3 = nn.Conv2d(in_channels=self.baseChannel, out_channels=self.baseChannel,
                                  kernel_size=3, stride=1, padding=1, bias=False, dilation=1)
                                #   kernel_size=3, stride=1, padding=5, bias=False, dilation=5, groups=self.baseChannel)
        

        self.p_conv_1 = nn.Conv2d(in_channels=self.baseChannel, out_channels=self.baseChannel,
                                  kernel_size=1, stride=1, padding=0, bias=False, dilation=1)
        self.p_conv_2 = nn.Conv2d(in_channels=self.baseChannel, out_channels=self.baseChannel,
                                  kernel_size=1, stride=1, padding=0, bias=False, dilation=1)
        self.p_conv_3 = nn.Conv2d(in_channels=self.baseChannel, out_channels=self.baseChannel,
                                  kernel_size=1, stride=1, padding=0, bias=False, dilation=1)
        

        self.b1_conv_4 = nn.Conv2d(in_channels=self.baseChannel, out_channels=self.baseChannel,
                                  kernel_size=3, stride=1, padding=1, bias=False, dilation=1, groups=self.baseChannel//opt.conv_g)
        self.b1_conv_5 = nn.Conv2d(in_channels=self.baseChannel, out_channels=self.baseChannel,
                                  kernel_size=3, stride=1, padding=1, bias=False, dilation=1, groups=self.baseChannel//opt.conv_g)
                                #   kernel_size=3, stride=1, padding=2, bias=False, dilation=2, groups=self.baseChannel)
        self.b1_conv_6 = nn.Conv2d(in_channels=self.baseChannel, out_channels=self.baseChannel,
                                  kernel_size=3, stride=1, padding=1, bias=False, dilation=1, groups=self.baseChannel//opt.conv_g)
                                #   kernel_size=3, stride=1, padding=5, bias=False, dilation=5, groups=self.baseChannel)
        

        self.p_conv_4 = nn.Conv2d(in_channels=self.baseChannel, out_channels=self.baseChannel,
                                  kernel_size=1, stride=1, padding=0, bias=False, dilation=1)
        self.p_conv_5 = nn.Conv2d(in_channels=self.baseChannel, out_channels=self.baseChannel,
                                  kernel_size=1, stride=1, padding=0, bias=False, dilation=1)
        self.p_conv_6 = nn.Conv2d(in_channels=self.baseChannel, out_channels=self.baseChannel,
                                  kernel_size=1, stride=1, padding=0, bias=False, dilation=1)
        
        self.output_Y = nn.Conv2d(
            in_channels=self.baseChannel, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
        # ### If load retrined model
        if opt.pre_train != '.':
            pt = torch.load(opt.pre_train)
            for p in pt.keys():
                if p != 'input_Y.weight' and p !='output_Y.weight':
                    self.state_dict()[p].copy_(pt[p])
                    # self.named_parameters()[p].requires_grad = False
                    for n,p2 in self.named_parameters():
                        if n == p:
                            p2.requires_grad = False
            # self.b1_conv_1.requires_grad = False
            # self.b1_conv_2.requires_grad = False
            # self.b1_conv_3.requires_grad = False
            # self.p_conv_1.requires_grad = False
            # self.p_conv_2.requires_grad = False
            # self.p_conv_3.requires_grad = False
        # ###

    def forward(self, x):
        Y_out = x[:, 0, :, :].unsqueeze(1)
        residual_Y = Y_out
        
        Y_out = self.input_Y(Y_out)
        Y_out = self.InstanceNorm(Y_out)
        Y_out = self.act(Y_out)

        yr = Y_out

        '''block1'''
        Y_out = self.b1_conv_1(Y_out)
        Y_out = self.InstanceNorm(Y_out)
        Y_out = self.p_conv_1(Y_out)
        Y_out = self.InstanceNorm(Y_out)
        Y_out = self.act(Y_out)

        Y_out = self.b1_conv_2(Y_out)
        Y_out = self.InstanceNorm(Y_out)
        Y_out = self.p_conv_2(Y_out)
        Y_out = self.InstanceNorm(Y_out)
        Y_out = self.act(Y_out)

        Y_out = self.b1_conv_3(Y_out)
        Y_out = self.InstanceNorm(Y_out)
        Y_out = self.p_conv_3(Y_out)
        Y_out = self.InstanceNorm(Y_out)
        Y_out = self.act(Y_out)
        
        Y_out = self.b1_conv_4(Y_out)
        Y_out = self.InstanceNorm(Y_out)
        Y_out = self.p_conv_4(Y_out)
        Y_out = self.InstanceNorm(Y_out)
        Y_out = self.act(Y_out)
        
        Y_out = self.b1_conv_5(Y_out)
        Y_out = self.InstanceNorm(Y_out)
        Y_out = self.p_conv_5(Y_out)
        Y_out = self.InstanceNorm(Y_out)
        Y_out = self.act(Y_out)

        Y_out = self.b1_conv_6(Y_out)
        Y_out = self.InstanceNorm(Y_out)
        Y_out = self.p_conv_6(Y_out)
        Y_out = self.InstanceNorm(Y_out)
        Y_out = self.act(Y_out)

        Y_out = torch.add(Y_out, yr)

        Y_out = self.output_Y(Y_out)
        Y_out = torch.add(Y_out, residual_Y)

        return Y_out
