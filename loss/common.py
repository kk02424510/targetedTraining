import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import option as opt


from math import exp

class R_loss(nn.Module):
    def __init__(self,codebook):
        super(R_loss, self).__init__()
        self.dist = self._make_dist()

    def forward(self,para):
        self.loss = self.cost2(para)
        return self.loss
    # def backward(self, grad_output):

    #     return(grad_output)

    def _make_dist(self,mean = 0):
        
        return (torch.distributions.Normal(0,1))

    def cost2(self ,para):
        total = 0
        for name, paramData in para():
            if paramData.requires_grad:
                # if paramData.size(3) != 1:
                # if name != 'output_Y':
                    pMax, pMin = torch.max(paramData), torch.min(paramData)
                    scale = max(abs(pMax),abs(pMin))/((1<<opt.modelQF-1) - 1)
                    quanWeight = paramData/scale
                    hi = self.dist.log_prob(torch.zeros_like(quanWeight))
                    log_porb = (hi-self.dist.log_prob(paramData)).sum()
                    total += log_porb
        return total


class L1_Charbonnier_loss(nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt( diff * diff + self.eps )
        loss = torch.mean(error) 
        return loss
        
class SSIM(nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self._create_window(window_size, self.channel)
    
    def forward(self, img1, img2):
        img1 = img1.view(1,1,img1.shape[-2],img1.shape[-1])
        img2 = img1.view(1,1,img2.shape[-2],img1.shape[-1])
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self._create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return self._ssim(img1, img2, window, self.window_size, channel, self.size_average)

    def _gaussian(self, window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def _create_window(self, window_size, channel):
        _1D_window = self._gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
        return window

    def _ssim(self, img1, img2, window, window_size, channel, size_average = True):
        mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
        mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2

        sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)