from importlib import import_module

import torch
import torch.nn as nn
import torch.nn.functional as F

import loss.common as common
import loss.vgg_perceptual_loss as p_loss
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
# import utils

class loss:
    def __init__(self, opt, codebook):
        self.codebook = codebook
        self.opt = opt

    # def get_loss(self):
    def get_loss(self):
        print('Preparing loss function...')

        my_loss = []
        losslist = self.opt.loss.split('+')
        for loss in losslist:
            weight, loss_type = loss.split('*')
            if loss_type == 'MSE':
                loss_function = nn.MSELoss()
                # loss_function = nn.MSELoss(reduction='none')
            elif loss_type == 'L1':
                loss_function = nn.L1Loss()
            elif loss_type == 'L1char':
                loss_function = common.L1_Charbonnier_loss()
            elif loss_type == 'SSIM':
                loss_function = SSIM(data_range=255, channel=1, nonnegative_ssim=True)
                # loss_function = common.SSIM()
            elif loss_type == 'MS-SSIM':
                loss_function = MS_SSIM(data_range=255, channel=1, nonnegative_ssim=True)
            elif loss_type == 'R_loss':
                loss_function = common.R_loss(self.codebook)
            elif loss_type == 'P_loss':
                loss_function = p_loss.VGGPerceptualLoss()
            else:
                continue
            
            # Enable cuda
            if not self.opt.no_cuda:
                print('\tCUDA is ready!')
                torch.cuda.manual_seed(self.opt.seed)
                loss_function = loss_function.cuda()
                if self.opt.n_GPUs > 1:
                    loss_function = nn.DataParallel(loss_function, range(0, self.opt.n_GPUs))

            # if weight == 'var' :
            #     my_loss.append({
            #         'type': loss_type,
            #         'weight': float(var),
            #         'function': loss_function})
            # else:
            my_loss.append({
                'type': loss_type,
                'weight': float(weight),
                'function': loss_function})

        if len(losslist) > 1:
            my_loss.append({
                'type': 'Total',
                'weight': 0,
                'function': None})

        print(my_loss)

        return my_loss
