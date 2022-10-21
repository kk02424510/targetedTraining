import math
import random
import numpy as np
from decimal import Decimal
from functools import reduce

import utils

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.utils as tu
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from model.binarized_modules import BinarizeLinear,BinarizeConv2d,BinaryActivation, HardBinaryConv,LearnableBias

from tensorboardX import SummaryWriter

torch.backends.cudnn.benchmark = True

class Trainer():
    def __init__(self, ckp, opt):
        self.opt = opt

        self.model, self.loss, self.optimizer, self.scheduler = ckp.load()
        self.ckp = ckp

        self.log_training = 0
        self.log_test = 0
        self.r_w = 0
        self.filter_mask = {}
        self.flag = False
        self.quanFlag = False
        # self.writer = SummaryWriter()
        self.visdomImageDict = {}

    def train(self):
        if 'step' in self.opt.decay_type:
            # self.scheduler.step()
            lr = self.scheduler.get_last_lr()[0]

        elif 'auto' in self.opt.decay_type:
            try:
                self.scheduler.step(self.best[0][0])
            except:
                self.scheduler.step(0)
                self.scheduler.last_epoch = 1
            lr = self.scheduler.optimizer.param_groups[0]['lr']

        epoch = self.scheduler.last_epoch
        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.ckp.add_log(torch.zeros(1, len(self.loss)))
        self.model.train()

        if self.flag == True:
            self._make_optimizer()
            self.flag = False

        timer_data, timer_model = utils.timer(), utils.timer()


        if self.opt.itercontrol:
            iter_time = self.inputData.shape[0]*self.opt.test_every
        else:
            iter_time = self.opt.test_every
            
        for batch in range(self.opt.test_every):
            if batch == 0:
                print(self.inputData.size())

            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()

            self.output = self.model(self.inputData)
            loss = self._calc_loss(self.output, self.target)

            loss.backward()

            if self.opt.clip:
                thres = self.opt.clip
                if self.opt.adjustable_clip:
                    thres = self.opt.clip/lr/10
                nn.utils.clip_grad_norm_(self.model.parameters(), thres)
            self.optimizer.step()

            timer_model.hold()

            if (batch + 1) % self.opt.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.opt.batch_size,
                    self.opt.batch_size * self.opt.test_every,
                    self._display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))
            timer_data.tic()
        self.scheduler.step()
        self.ckp.log_training[-1, :] /= self.opt.batch_size * self.opt.test_every

    def test(self, save=True):
        epoch = self.scheduler.last_epoch

        if save:
            self.ckp.write_log('\nEvaluation:')
            self.ckp.add_log(torch.zeros(1, 12), False)
        self.model.eval()

        timer_test = utils.timer()
        set_name = self.opt.dataset
        ori_acc, eval_acc = 0, 0

        with torch.no_grad():
                self.output = self.model(self.inputData)

        ori_acc = utils.calc_PSNR(
            self.inputData, self.target, 0)
        eval_acc = utils.calc_PSNR(
            self.output, self.target, 0)

        self.ckp.log_test[-1, 0] = eval_acc

        # if eval_acc != 'inf':
        #     self.ckp.log_test[-1, 0] = eval_acc
        # else:
        #     self.ckp.log_test[-1, 0] = 0

        ori_testlog = ori_acc
        if save:
            self.best = self.ckp.log_test.max(0)

        # if ori_acc =='inf':
        #     performance = 'Y Preprocessing: {}\tPSNR: {:.3f}\tSSIM: {:.4f}'.format(
        #         ori_acc, eval_acc, float(ssim(self.output, self.target)))

        # else:
        performance = 'Y Preprocessing: {:.3f}\tPSNR: {:.3f}\tSSIM: {:.4f}'.format(
            ori_acc, eval_acc, float(ssim(self.output, self.target)))

        if save:
            self.ckp.write_log(
                '[{}]\t{} (Best: {:.3f} from epoch {})'.format(
                    set_name,
                    performance,
                    self.best[0][0],
                    self.best[1][0] + 1,
                    ))
            self.ckp.write_log(
                'Time: {:.2f}s\n'.format(timer_test.toc()), refresh=True)
        else:
            print(performance)

        torch.cuda.empty_cache() #用來釋放valid後沒釋放的記憶體


    def _prepare(self, input, target, volatile=False):
        if not self.opt.no_cuda:
            input = input.cuda()
            target = target.cuda()

        input.requires_grad = not volatile
        target.requires_grad = not volatile

        # return input, target
        return input[:,0,:,:].view((1, 1, input.shape[-2], input.shape[-1])), target[:,0,:,:].view((1, 1, target.shape[-2], target.shape[-1]))

    def _prepare_v2(self, input, target, volatile=False):
        if not self.opt.no_cuda:
            input = input.cuda()
            target = target.cuda()

        input.requires_grad = not volatile
        target.requires_grad = not volatile

        # return input, target
        return input, target

    def _calc_loss(self, output, target):
        loss_list = []
        loss_func = []
        for i, l in enumerate(self.loss):
            loss_func.append(l['type'])
            if l['function'] == None:
                continue
            if isinstance(output, list):
                if isinstance(target, list):
                    loss = l['function'](output[i], target[i])
                else:
                    loss = l['function'](output[i], target)
            else:
                if self.opt.multi:
                    loss = l['function'](output[:,0,:,:], target[:,0,:,:]) + \
                            0.25*l['function'](output[:,1,:,:], target[:,1,:,:]) + \
                            0.25*l['function'](output[:,2,:,:], target[:,2,:,:])
                else:
                    if l['type'] == 'R_loss':
                        loss = l['function'](self.model.named_parameters)
                    elif l['type'] == 'P_loss':
                        loss = l['function'](output, target, feature_layers = [0,1])
                    else:
                        loss = l['function'](output[:,0,:,:], target[:,0,:,:])

            if 'SSIM' in l['type']:
                loss_list.append(l['weight'] * (1 - loss))
            else:
                if l['weight'] == 'var':
                    loss_list.append(self.r_w * loss)
                else:
                    loss_list.append(l['weight'] * loss)
            # self.ckp.log_training[-1, i] += loss.data[0]  #0.3
            
            self.ckp.log_training[-1, i] += loss.item()     #0.4~
        if 'MSE' in loss_func and 'R_loss' in loss_func:
            MSE_n = loss_func.index('MSE')
            R_loss_n = loss_func.index('R_loss')
            while loss_list[MSE_n] < abs(loss_list[R_loss_n]):
                loss_list[R_loss_n] *= 0.1

        loss_total = reduce((lambda x, y: x + y), loss_list)
        if len(self.loss) > 1:
            # self.ckp.log_training[-1, -1] += loss_total.data[0]
            self.ckp.log_training[-1, -1] += loss_total.item()

        return loss_total

    def _display_loss(self, batch):
        log = [
            '[{}: {:.4f}] '.format(t['type'], l*(255 / self.opt.rgb_range) / (batch + 1)) for l, t in zip(self.ckp.log_training[-1], self.loss)
        ]
        return ''.join(log)

    def quantize_weight(self, show = True):
        ''' fix-point quantization without finetune '''
        for layer, (name, module) in enumerate(self.model._modules.items()):
            if isinstance(module, nn.modules.Conv2d):
                rawParam = module.weight.data.cpu()
                pMax, pMin = float(np.max(np.array(rawParam))), float(np.min(np.array(rawParam)))
                module.interval = 2 * max(abs(pMax),abs(pMin)) / ((1<<self.opt.modelQF) - 1) + 0.001
                module.mean = 0
                module.quanWeight = torch.round((rawParam - module.mean) / module.interval)
                module.weight.data.copy_(module.quanWeight * module.interval + module.mean)
        ''' 量化後的結果 '''
        if show: self.test(False)

    def quantize_weight_v3(self, show = True):
        ''' fix-point quantization without finetune '''
        for layer, (name, module) in enumerate(self.model._modules.items()):
            if isinstance(module, nn.modules.Conv2d):
                rawParam = module.weight.data.cpu()
                pMax, pMin = float(np.max(np.array(rawParam))), float(np.min(np.array(rawParam)))
                # module.interval = 2 * max(abs(pMax),abs(pMin)) / ((1<<self.opt.modelQF) - 1) + 0.001
                module.interval = max(abs(pMax),abs(pMin))/((1<<self.opt.modelQF-1) - 1)
                qparam = torch.quantize_per_tensor(rawParam, module.interval,0,dtype=torch.qint8)
                module.quanWeight = qparam.int_repr()
                module.weight.data.copy_(qparam.dequantize())
        ''' 量化後的結果 '''
        if show: self.test(False)
    
    def quantize_weight_v4(self, show = True):
        for layer, (name, module) in enumerate(self.model._modules.items()):
            if isinstance(module, nn.modules.Conv2d):
                if module.weight.requires_grad:
                    rawParam = module.weight.data.cpu()
                    pMax, pMin = float(np.max(np.array(rawParam))), float(np.min(np.array(rawParam)))
                    # module.interval = 2 * max(abs(pMax),abs(pMin)) / ((1<<self.opt.modelQF) - 1) + 0.001
                    module.interval = max(abs(pMax),abs(pMin))/((1<<self.opt.modelQF-1) - 1)
                    qparam = torch.quantize_per_tensor(rawParam, module.interval,0,dtype=torch.qint8)
                    module.quanWeight = qparam.int_repr()
                    module.weight.data.copy_(qparam.dequantize())
        ''' 量化後的結果 '''
        if show: self.test(False)
    
    def quantize_weight_pt(self, show = True):
        ''' fix-point quantization without finetune '''
        for layer, (name, module) in enumerate(self.model._modules.items()):
            if name !='pre_trained':
                if isinstance(module, nn.modules.Conv2d):
                    rawParam = module.weight.data.cpu()
                pMax, pMin = float(np.max(np.array(rawParam))), float(np.min(np.array(rawParam)))
                module.interval = (pMax-pMin)/(qmax-qmin)
                module.mean = 0
                qparam = torch.quantize_per_tensor(rawParam, module.interval,0,dtype=torch.qint8)
                module.quanWeight = qparam.int_repr()
                module.weight.data.copy_(qparam.dequantize())
        ''' 量化後的結果 '''
        if show: self.test(False)

    def quantize_weight_v2(self, show = True):
        ''' fix-point quantization without finetune '''
        for layer, (name, module) in enumerate(self.model._modules.items()):
  
            for l in module:
                if isinstance(l, nn.modules.Conv2d):
                    rawParam = l.weight.data.cpu()
                    pMax, pMin = float(np.max(np.array(rawParam))), float(np.min(np.array(rawParam)))
                    l.interval = 2 * max(abs(pMax),abs(pMin)) / ((1<<self.opt.modelQF) - 1) + 0.001
                    l.mean = 0
                    l.quanWeight = torch.round((rawParam - l.mean) / l.interval)
                    l.weight.data.copy_(l.quanWeight * l.interval + l.mean)
        ''' 量化後的結果 '''
        if show: self.test(False)

    def quantize_weight_HB(self, show = True):
        ''' fix-point quantization without finetune '''
        for layer, (name, module) in enumerate(self.model._modules.items()):
            for l in module:
                if isinstance(l, nn.modules.Conv2d):
                    rawParam = l.weight.data.cpu()
                    pMax, pMin = float(np.max(np.array(rawParam))), float(np.min(np.array(rawParam)))
                    l.interval = 2 * max(abs(pMax),abs(pMin)) / ((1<<self.opt.modelQF) - 1) + 0.001
                    l.mean = 0
                    l.quanWeight = torch.round((rawParam - l.mean) / l.interval)
                    l.weight.data.copy_(l.quanWeight * l.interval + l.mean)
                elif isinstance(l, HardBinaryConv):
                    rawParam = l.binary_weights.cpu()
                    # print(rawParam)
                    # l.kernel_symbol = np.max(np.array(rawParam),(2,3))
                    m = nn.MaxPool2d(3)
                    l.kernel_symbol = m(rawParam)
                    # print(nr)
                    # print(np[:].max())
                    # pMax, pMin = float(np.max(np.array(rawParam))), float(np.min(np.array(rawParam)))
                    # l.interval = 2 * max(abs(pMax),abs(pMin)) / ((1<<self.opt.modelQF) - 1) + 0.001
                    # l.mean = 0
                    l.quanWeight = (rawParam > 0).int()
                    l.binary_weights.copy_((l.quanWeight - 0.5) * 2 *l.kernel_symbol  )

        ''' 量化後的結果 '''
        if show: self.test(False)

    def quantize_weight_HB_vgg(self, show = True):
        ''' fix-point quantization without finetune '''
        for layer, (name, module) in enumerate(self.model._modules.items()):
            if name !='vgg16':
                for l in module:
                    if isinstance(l, nn.modules.Conv2d):
                        rawParam = l.weight.data.cpu()
                        pMax, pMin = float(np.max(np.array(rawParam))), float(np.min(np.array(rawParam)))
                        l.interval = 2 * max(abs(pMax),abs(pMin)) / ((1<<self.opt.modelQF) - 1) + 0.001
                        l.mean = 0
                        l.quanWeight = torch.round((rawParam - l.mean) / l.interval)
                        l.weight.data.copy_(l.quanWeight * l.interval + l.mean)
                    elif isinstance(l, HardBinaryConv):
                        rawParam = l.binary_weights.cpu()
                        m = nn.MaxPool2d(3)
                        l.kernel_symbol = m(rawParam)
                        l.quanWeight = (rawParam > 0).int()
                        l.binary_weights.copy_((l.quanWeight - 0.5) * 2 *l.kernel_symbol  )

        ''' 量化後的結果 '''
        if show: self.test(False)

    def terminate(self):
        epoch = self.scheduler.last_epoch
        return epoch >= self.opt.epochs

    def save(self,name,num):
        torch.save(self.model.state_dict(), './model/{}/{}_para_{}.pt'.format(self.opt.save,name,num))