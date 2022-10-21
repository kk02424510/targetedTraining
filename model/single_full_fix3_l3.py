# python main.py --model Multi_04 --colortype yuv --loss 1*MSE --clip 0.4 --lr 0.1 --lr_decay 40 --gamma 0.1 --batch_size 64 --patch_size 80 --test_every 2500 --print_every 250 --epochs 160 --optimizer SGD --weight_decay 1e-4 --save_result --rgb_range 255
# python main.py --model Multi_03-ev2 --colortype yuv --loss 1*L1char --clip 0.4 --lr 0.01 --lr_decay 40 --gamma 0.1 --batch_size 32 --patch_size 80 --test_every1250 --print_every 125 --epochs 120 --optimizer SGD --weight_decay 1e-4 --save_result --multi --qp 10 --data_train DIV2K --n_train 900 --save qp10
# Base CNN Model + Recursive 3 times + residual block + dilated
import sys
import torch
import torch.nn as nn
from math import sqrt
from model import common
from option import opt
import torchvision.models as models

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

# class vgg16(nn.Module):
#     def __init__(self):
#         super(vgg16, self).__init__()
#         model = models.vgg16(pretrained=True)

#         self.model = nn.Sequential(*list(model.children())[:3])
#     def forward(self,x):
#         y = self.model(x)
#         return y

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # model = models.resnet18(pretrained=True)
        model = models.vgg16(pretrained=True)
        self.conv_g = opt.conv_g
        # set ft[:3] conv1 (64)
        # self.pre_trained = nn.Sequential(*list(model.children())[:-1])
        self.pre_trained = nn.Sequential(*list(model.features)[:opt.vgg_l]).eval()
        for p in self.parameters():
            p.requires_grad = False
        self.baseChannel = 16
        self.input_Y = nn.Conv2d(
            in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        self.act = nn.ReLU()
        # self.IN = nn.InstanceNorm2d(self.baseChannel) 
        self.InstanceNorm = nn.InstanceNorm2d(self.baseChannel) 
        # self.IN1 = nn.InstanceNorm2d(self.baseChannel) 

        # self.conv1x1_1 = nn.Conv2d(in_channels=64, out_channels=32,
        #                           kernel_size=1, stride=1, padding=0, bias=False, dilation=1, groups=(32//self.conv_g))
        self.conv1 = nn.Conv2d(in_channels=self.baseChannel, out_channels=self.baseChannel,
                                  kernel_size=3, stride=1, padding=1, bias=False, dilation=1, groups=self.baseChannel)
        # self.IN1 = nn.InstanceNorm2d(3) 
        self.conv1x1 = nn.Conv2d(in_channels=self.baseChannel, out_channels=3,
                                  kernel_size=1, stride=1, padding=0, bias=False, dilation=1)
        self.conv2 = nn.Conv2d(in_channels=self.baseChannel, out_channels=self.baseChannel,
                                  kernel_size=3, stride=1, padding=1, bias=False, dilation=1, groups=self.baseChannel)
                                #   kernel_size=3, stride=1, padding=2, bias=False, dilation=2, groups=self.baseChannel)
        # self.IN2 = nn.InstanceNorm2d(16) 
        # self.conv1x1_3 = nn.Conv2d(in_channels=16, out_channels=8,
        #                           kernel_size=1, stride=1, padding=0, bias=False, dilation=1, groups=(8//self.conv_g))
        self.conv3 = nn.Conv2d(in_channels=self.baseChannel, out_channels=self.baseChannel,
                                  kernel_size=3, stride=1, padding=1, bias=False, dilation=1, groups=self.baseChannel)
        # self.b1_conv_1 = nn.Conv2d(in_channels=self.baseChannel, out_channels=self.baseChannel,
        #                           kernel_size=3, stride=1, padding=1, bias=False, dilation=1, groups=self.baseChannel)
        # self.b1_conv_2 = nn.Conv2d(in_channels=self.baseChannel, out_channels=self.baseChannel,
        #                           kernel_size=3, stride=1, padding=1, bias=False, dilation=1, groups=self.baseChannel)
        #                         #   kernel_size=3, stride=1, padding=2, bias=False, dilation=2, groups=self.baseChannel)
        # self.b1_conv_3 = nn.Conv2d(in_channels=self.baseChannel, out_channels=self.baseChannel,
        #                           kernel_size=3, stride=1, padding=1, bias=False, dilation=1, groups=self.baseChannel)
        #                         #   kernel_size=3, stride=1, padding=5, bias=False, dilation=5, groups=self.baseChannel)
        

        self.p_conv_1 = nn.Conv2d(in_channels=self.baseChannel, out_channels=self.baseChannel,
                                  kernel_size=1, stride=1, padding=0, bias=False, dilation=1)
        self.p_conv_2 = nn.Conv2d(in_channels=self.baseChannel, out_channels=self.baseChannel,
                                  kernel_size=1, stride=1, padding=0, bias=False, dilation=1)
        self.p_conv_3 = nn.Conv2d(in_channels=self.baseChannel, out_channels=self.baseChannel,
                                  kernel_size=1, stride=1, padding=0, bias=False, dilation=1)
        self.pixel_shuffle = nn.PixelShuffle(4)
        # self.pixel_shuffle2 = nn.PixelShuffle(2)
        
        # self.b1_conv_1 = nn.Conv2d(in_channels=self.baseChannel, out_channels=self.baseChannel,
        #                           kernel_size=3, stride=1, padding=1, bias=False, dilation=1, groups=self.baseChannel)
        # self.b1_conv_2 = nn.Conv2d(in_channels=self.baseChannel, out_channels=self.baseChannel,
        #                           kernel_size=3, stride=1, padding=1, bias=False, dilation=1, groups=self.baseChannel)
        #                         #   kernel_size=3, stride=1, padding=2, bias=False, dilation=2, groups=self.baseChannel)
        # self.b1_conv_3 = nn.Conv2d(in_channels=self.baseChannel, out_channels=self.baseChannel,
        #                           kernel_size=3, stride=1, padding=1, bias=False, dilation=1, groups=self.baseChannel)
        #                         #   kernel_size=3, stride=1, padding=5, bias=False, dilation=5, groups=self.baseChannel)
        

        # self.p_conv_1 = nn.Conv2d(in_channels=self.baseChannel, out_channels=self.baseChannel,
        #                           kernel_size=1, stride=1, padding=0, bias=False, dilation=1)
        # self.p_conv_2 = nn.Conv2d(in_channels=self.baseChannel, out_channels=self.baseChannel,
        #                           kernel_size=1, stride=1, padding=0, bias=False, dilation=1)
        # self.p_conv_3 = nn.Conv2d(in_channels=self.baseChannel, out_channels=self.baseChannel,
        #                           kernel_size=1, stride=1, padding=0, bias=False, dilation=1)
        # self.conv1x1_4 = nn.Conv2d(in_channels=8, out_channels=4,
        #                           kernel_size=1, stride=1, padding=0, bias=False, dilation=1, groups=4)
        self.output_Y = nn.Conv2d(
            in_channels=6, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)

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
        residual_Y = Y_out
        Y_out = self.input_Y(Y_out)
        yr = Y_out
        # Y_out = torch.cat((Y_out,Y_out,Y_out),1)
        # Y_out = self.input_Y(Y_out)
        # Y_out = self.InstanceNorm(Y_out)
        # Y_out = self.act(Y_out)
        # yr = Y_out
        Y_out = self.pre_trained(Y_out)
        Y_out = self.pixel_shuffle(Y_out)
        # yr = Y_out

        '''block1'''
        # Y_out = self.conv1x1_1(Y_out)
        # Y_out = self.IN1(Y_out)
        Y_out = self.conv1(Y_out)
        Y_out = self.InstanceNorm(Y_out)
        Y_out = self.p_conv_1(Y_out)
        Y_out = self.InstanceNorm(Y_out)
        Y_out = self.act(Y_out)

        # Y_out = self.conv1x1_2(Y_out)
        # Y_out = self.IN2(Y_out)
        Y_out = self.conv2(Y_out)
        Y_out = self.InstanceNorm(Y_out)
        Y_out = self.p_conv_2(Y_out)
        Y_out = self.InstanceNorm(Y_out)
        Y_out = self.act(Y_out)

        # Y_out = self.conv1x1_3(Y_out)
        # Y_out = self.IN3(Y_out)
        Y_out = self.conv3(Y_out)
        Y_out = self.InstanceNorm(Y_out)
        Y_out = self.p_conv_3(Y_out)
        Y_out = self.InstanceNorm(Y_out)
        Y_out = self.act(Y_out)
            
        Y_out = self.conv1x1(Y_out)
        # Y_out = self.IN1(Y_out)
        # Y_out = self.b1_conv_1(Y_out)
        # Y_out = self.InstanceNorm(Y_out)
        # Y_out = self.p_conv_1(Y_out)
        # Y_out = self.InstanceNorm(Y_out)
        # Y_out = self.act(Y_out)

        # Y_out = self.b1_conv_2(Y_out)
        # Y_out = self.InstanceNorm(Y_out)
        # Y_out = self.p_conv_2(Y_out)
        # Y_out = self.InstanceNorm(Y_out)
        # Y_out = self.act(Y_out)

        # Y_out = self.b1_conv_3(Y_out)
        # Y_out = self.InstanceNorm(Y_out)
        # Y_out = self.p_conv_3(Y_out)
        # Y_out = self.InstanceNorm(Y_out)
        # Y_out = self.act(Y_out)

        Y_out = torch.cat((Y_out, yr),1)
        # Y_out = torch.add(Y_out,yr)
        # Y_out = self.conv1x1_4(Y_out)
        # Y_out = self.pixel_shuffle2(Y_out)
        Y_out = self.output_Y(Y_out)
        Y_out = torch.add(Y_out, residual_Y)

        return Y_out
