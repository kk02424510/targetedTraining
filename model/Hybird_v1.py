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
import torch.nn.functional as F
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

class BinaryActivation(nn.Module):
    def __init__(self):
        super(BinaryActivation, self).__init__()

    def forward(self, x):
        out_forward = torch.sign(x)
        mask1 = x < -1
        mask2 = x < 0
        mask3 = x < 1
        out1 = (-1) * mask1.type(torch.float32) + (x*x + 2*x) * (1-mask1.type(torch.float32))
        out2 = out1 * mask2.type(torch.float32) + (-x*x + 2*x) * (1-mask2.type(torch.float32))
        out3 = out2 * mask3.type(torch.float32) + 1 * (1- mask3.type(torch.float32))
        out = out_forward.detach() - out3.detach() + out3

        return out
        
class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1,out_chn,1,1), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out

class HardBinaryConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(HardBinaryConv, self).__init__()
        self.stride = stride
        self.padding = padding
        self.number_of_weights = in_channels * out_channels * kernel_size * kernel_size
        self.shape = (out_channels, in_channels, kernel_size, kernel_size)
        self.weights = nn.Parameter(torch.rand((self.number_of_weights,1)) * 0.001, requires_grad=True)

    def forward(self, x):
        real_weights = self.weights.view(self.shape)
        scaling_factor = torch.mean(torch.mean(torch.mean(abs(real_weights),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
        #print(scaling_factor, flush=True)
        scaling_factor = scaling_factor.detach()
        binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        #print(binary_weights, flush=True)
        y = F.conv2d(x, binary_weights, stride=self.stride, padding=self.padding)

        return y

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

        # self.hb_block1 = nn.Sequential(
        #     BinaryActivation(),
        #     HardBinaryConv(
        #     in_channels=self.baseChannel, out_channels=self.baseChannel*5, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(self.baseChannel*5),
            
        #     BinaryActivation(),
        #     HardBinaryConv(
        #     in_channels=self.baseChannel*5, out_channels=self.baseChannel*3, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(self.baseChannel*3),
            
        #     BinaryActivation(),
        #     HardBinaryConv(
        #     in_channels=self.baseChannel*3, out_channels=self.baseChannel, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(self.baseChannel)
            
        # )

        # self.hb_block2 = nn.Sequential(
        #     BinaryActivation(),
        #     HardBinaryConv(
        #     in_channels=self.baseChannel, out_channels=self.baseChannel, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(self.baseChannel),
            
        #     BinaryActivation(),
        #     HardBinaryConv(
        #     in_channels=self.baseChannel, out_channels=self.baseChannel, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(self.baseChannel),
            
        #     BinaryActivation(),
        #     HardBinaryConv(
        #     in_channels=self.baseChannel, out_channels=self.baseChannel, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(self.baseChannel)
            
        # )

        # self.block2 = nn.Sequential(
        #     BinaryActivation(),
        #     HardBinaryConv(
        #     in_channels=self.baseChannel, out_channels=self.baseChannel*4, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(self.baseChannel*4),
            
        #     BinaryActivation(),
        #     HardBinaryConv(
        #     in_channels=self.baseChannel*4, out_channels=self.baseChannel*2, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(self.baseChannel*2),
            
        #     BinaryActivation(),
        #     HardBinaryConv(
        #     in_channels=self.baseChannel*2, out_channels=self.baseChannel, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(self.baseChannel)
            
        # )

        # self.block2 = nn.Sequential(
        #     BinaryActivation(),
        #     HardBinaryConv(
        #     in_channels=1, out_channels=self.baseChannel*3, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(self.baseChannel*3),
            
        #     BinaryActivation(),
        #     HardBinaryConv(
        #     in_channels=self.baseChannel*3, out_channels=self.baseChannel, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(self.baseChannel),
            
        #     BinaryActivation(),
        #     HardBinaryConv(
        #     in_channels=self.baseChannel, out_channels=1, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(1)
        # )

        self.og_block = nn.Sequential(
            
            BinarizeConv2d(
            in_channels=self.baseChannel, out_channels=self.baseChannel, kernel_size=3, stride=1, padding=1, bias=False, groups=self.baseChannel),
            nn.BatchNorm2d(self.baseChannel),
            nn.Hardtanh(inplace=True),

            # nn.Hardtanh(inplace=True),
            BinarizeConv2d(
            in_channels=self.baseChannel, out_channels=self.baseChannel, kernel_size=3, stride=1, padding=1, bias=False, groups=self.baseChannel),
            nn.BatchNorm2d(self.baseChannel),
            nn.Hardtanh(inplace=True),

            # nn.Hardtanh(inplace=True),
            BinarizeConv2d(
            in_channels=self.baseChannel, out_channels=self.baseChannel, kernel_size=3, stride=1, padding=1, bias=False, groups=self.baseChannel),
            nn.BatchNorm2d(self.baseChannel),
            nn.Hardtanh(inplace=True)
        )

        # self.og_block2 = nn.Sequential(
        #     BinaryActivation(),
        #     BinarizeConv2d(
        #     in_channels=self.baseChannel, out_channels=self.baseChannel, kernel_size=3, stride=1, padding=1, bias=True, groups=self.baseChannel),
        #     nn.BatchNorm2d(self.baseChannel),

        #     BinaryActivation(),
        #     BinarizeConv2d(
        #     in_channels=self.baseChannel, out_channels=self.baseChannel, kernel_size=3, stride=1, padding=1, bias=True, groups=self.baseChannel),
        #     nn.BatchNorm2d(self.baseChannel),
        #     # nn.Hardtanh(inplace=True),

        #     BinaryActivation(),
        #     BinarizeConv2d(
        #     in_channels=self.baseChannel, out_channels=self.baseChannel, kernel_size=3, stride=1, padding=1, bias=True, groups=self.baseChannel),
        #     nn.BatchNorm2d(self.baseChannel),
        #     # nn.Hardtanh(inplace=True)
        # )

        # self.og_block2 = nn.Sequential(
        #     BinaryActivation(),
        #     BinarizeConv2d(
        #     in_channels=self.baseChannel, out_channels=self.baseChannel*4, kernel_size=3, stride=1, padding=1, bias=True, groups=self.baseChannel),
        #     nn.BatchNorm2d(self.baseChannel*4),

        #     BinaryActivation(),
        #     BinarizeConv2d(
        #     in_channels=self.baseChannel*4, out_channels=self.baseChannel*2, kernel_size=3, stride=1, padding=1, bias=True, groups=self.baseChannel),
        #     nn.BatchNorm2d(self.baseChannel*2),
        #     # nn.Hardtanh(inplace=True),

        #     BinaryActivation(),
        #     BinarizeConv2d(
        #     in_channels=self.baseChannel*2, out_channels=self.baseChannel, kernel_size=3, stride=1, padding=1, bias=True, groups=self.baseChannel),
        #     nn.BatchNorm2d(self.baseChannel),
        #     # nn.Hardtanh(inplace=True)
        # )

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
        Y_out = self.og_block(Y_out)
        # Y_out = torch.add(Y_out,yr)
        # yr = Y_out.clone()
        # Y_out = self.og_block2(Y_out)
        # Y_out = torch.add(Y_out,yr)
        # Y_out = torch.add(Y_out,yr)
        # yr = Y_out.clone()
        # Y_out = self.block3(Y_out)
        # Y_out = torch.add(Y_out,yr)
        Y_out = torch.cat((Y_out,yr),1)
        Y_out = self.output_Y(Y_out)
        Y_out = torch.add(Y_out, residual_Y)
        
        return Y_out
