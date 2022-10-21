import os 
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn.preprocessing import minmax_scale
from option import opt

def make_model():
    model = models.vgg16(pretrained=True).features[:opt.vgg_l]
    # model = models.vgg16(pretrained=True)
    model = model.eval()
    model.cuda()
    return model

def extract_feature(model, img):
    model.eval()

    if img.shape[1] != 3:
        img = img.repeat(1,3,1,1)
    img = img.cuda()

    res = model(Variable(img))
    res = channel_attention(res)
    res_list = res.data.cpu().tolist()[0]
    # res_nor = minmax_scale(res_list,feature_range=(0, 1),axis = 0)
    
    return res_list


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

def channel_attention(x):
    model = ChannelGate(x.shape[1])
    # model = model.eval()
    model.cuda()

    res = model(Variable(x))

    return res

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        channel_att_sum = torch.sigmoid( channel_att_sum )
        # scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return channel_att_sum

def load(patchPairList,hPatchNum,wPatchNum):
    model = make_model()
    ft = []
    for num in range(hPatchNum*wPatchNum):
        ft.append(extract_feature(model,patchPairList[num][0]))

    # ft = minmax_scale(ft,feature_range=(0, 1),axis = 0)
    # sc = MinMaxScaler(feature_range=(0,1), copy=False)
    # ft_n = sc.fit(ft)
    torch.cuda.empty_cache()
    return ft