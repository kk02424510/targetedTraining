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
    #使用效益不大，原因是channel shuffle是依靠group conv的組數去進行通道洗牌
    #但在我們的case上所使用的group conv皆為分成1組
    #故一組做通道洗牌沒有意義。
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

class FeatureExtractor(nn.Module):  
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor,self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers
 
    def forward(self, x):
        outputs = []
        size = x.size(3)
        for name, module in self.submodule._modules.items():
            x = module(x)
            if name in self.extracted_layers:
                t = size // x.size(3)
                pixel_shuffle = nn.PixelShuffle(t)
                x_2 = pixel_shuffle(x)
        #         outputs += [x_2]
        # return outputs
                if len(outputs) == 0:
                    outputs = x_2
                else: 
                    outputs = torch.cat((outputs,x_2),1)
        return outputs


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        feature_extract_model = models.vgg16(pretrained=True)
        self.conv_g = opt.conv_g
        # exact_list=['3']
        # self.pre_trained = FeatureExtractor(nn.Sequential(*list(model.features)).eval(), exact_list)
        self.pre_trained = nn.Sequential(*list(feature_extract_model.features)[:opt.vgg_l]).eval()
        for p in self.parameters():
            p.requires_grad = False

        self.baseChannel = opt.ch
        # self.input_Y = nn.Conv2d(
        #     in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        self.act = nn.ReLU()
        # self.IN = nn.InstanceNorm2d(self.baseChannel) 
        self.InstanceNorm = nn.InstanceNorm2d(self.baseChannel) 
        # self.IN0 = nn.InstanceNorm2d(self.baseChannel) 
        # self.IN1 = nn.InstanceNorm2d(self.baseChannel) 
        # self.IN2 = nn.InstanceNorm2d(self.baseChannel)
        # self.IN4 = nn.InstanceNorm2d(self.baseChannel*2) 

        self.conv1x1_1 = nn.Conv2d(in_channels=self.baseChannel, out_channels=self.baseChannel,
                                  kernel_size=1, stride=1, padding=0, bias=False, dilation=1)
        self.conv3x3_1 = nn.Conv2d(in_channels=self.baseChannel, out_channels=self.baseChannel,
                                  kernel_size=3, stride=1, padding=1, bias=False, dilation=1, groups=self.baseChannel//opt.conv_g)
        
        self.conv1x1_2 = nn.Conv2d(in_channels=self.baseChannel, out_channels=self.baseChannel,
                                  kernel_size=1, stride=1, padding=0, bias=False, dilation=1,)
        self.conv3x3_2 = nn.Conv2d(in_channels=self.baseChannel, out_channels=self.baseChannel,
                                  kernel_size=3, stride=1, padding=1, bias=False, dilation=1, groups=self.baseChannel//opt.conv_g)

        self.conv1x1_3 = nn.Conv2d(in_channels=self.baseChannel, out_channels=self.baseChannel,
                                  kernel_size=1, stride=1, padding=0, bias=False, dilation=1)
        self.conv3x3_3 = nn.Conv2d(in_channels=self.baseChannel, out_channels=self.baseChannel,
                                  kernel_size=3, stride=1, padding=1, bias=False, dilation=1, groups=self.baseChannel//opt.conv_g)

        self.conv1x1 = nn.Conv2d(in_channels=64, out_channels=self.baseChannel,
                                  kernel_size=1, stride=1, padding=0, bias=False, dilation=1)
        
        # self.conv1x1_4 = nn.Conv2d(in_channels=self.baseChannel, out_channels=3,
        #                           kernel_size=1, stride=1, padding=0, bias=False, dilation=1)
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
                self.state_dict()[p].copy_(pt[p])
        # ###

    def forward(self, x):
        Y_out = x[:, 0, :, :].unsqueeze(1)
        residual_Y = Y_out
        '''Feature Extract Layer'''
        Y_out = torch.cat((residual_Y,residual_Y,residual_Y),1)
        Y_out = self.pre_trained(Y_out)

        '''compression channel'''
        Y_out = self.conv1x1(Y_out)
        Y_out = self.InstanceNorm(Y_out)
        yr = Y_out

        '''block1'''
        Y_out = self.conv1x1_1(Y_out)
        Y_out = self.InstanceNorm(Y_out)
        Y_out = self.conv3x3_1(Y_out)
        Y_out = self.InstanceNorm(Y_out)
        Y_out = self.act(Y_out)

        Y_out = self.conv1x1_2(Y_out)
        Y_out = self.InstanceNorm(Y_out)
        Y_out = self.conv3x3_2(Y_out)
        Y_out = self.InstanceNorm(Y_out)
        Y_out = self.act(Y_out)

        Y_out = self.conv1x1_3(Y_out)
        Y_out = self.InstanceNorm(Y_out)
        Y_out = self.conv3x3_3(Y_out)
        Y_out = self.InstanceNorm(Y_out)
        Y_out = self.act(Y_out)
            
        # Y_out = self.conv1x1_4(Y_out)
        # Y_out = self.IN4(Y_out)
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

        # Y_out = torch.cat((Y_out, yr),1)
        Y_out = torch.add(Y_out,yr)
        Y_out = self.output_Y(Y_out)
        Y_out = torch.add(Y_out, residual_Y)

        return Y_out
