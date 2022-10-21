import os
import math
import random
import time
import datetime
from functools import reduce

import csv
import numpy as np
import skimage.io as sio
import skimage.color as sc

import cv2
import PIL
import glymur
import scipy.misc as misc
from decimal import Decimal

from option import opt
from model import model
from loss import loss

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import torchvision.utils as tu

def quantize(img):
    if opt.rgb_range == 255:
        return img.clamp(0, 255).add(0.5).floor().div(255)
    elif opt.rgb_range == 1:
        return img.clamp(0,1)

def rgb2ycbcrT(rgb):
    rgb = rgb.numpy().transpose(1, 2, 0)
    yCbCr = sc.rgb2ycbcr(rgb) / 255

    return torch.Tensor(yCbCr[:, :, 0])

def calc_PSNR(input, target, channel):
    (_, c, h, w) = input.size()
    input = quantize(input.data[0])
    target = quantize(target.data[0])

    diff = None
    if opt.colortype == 'yuv':
        diff = input[channel,:,:].cpu() - target[channel,:,:].cpu()
    elif opt.colortype == 'rgb':
        diff = rgb2ycbcrT(input.cpu()) - rgb2ycbcrT(target.cpu())

    mse = diff.pow(2).mean()
    psnr = 20 * np.log10( 1. / math.sqrt(mse) )

    return psnr

def dTob(n, rawDataBit=18):
    '''
    把十進制的浮點數n轉換成二進制
    小數點後面保留pre位小數
    '''
    pre = rawDataBit - 3

    if n == 0:
        return '0'*rawDataBit

    sign_b = '0' if n>=0 else '1'
    n = abs(n)
    string_number1 = '{:.32f}'.format(n) #number1 表示十進制數，number2表示二進制數
    floatFlag = False
    if '.' in string_number1: #判斷是否含小數部分
        floatFlag = True

    string_integer, string_decimal = string_number1.split('.') #分離整數部分和小數部分
    integer = int(string_integer)
    decimal = Decimal(str(n)) - integer
    l1 = [0,1]
    l2 = []
    decimal_convert = ""
    string_integer2 = bin(integer)[2:]
    if floatFlag:
        i = 0
        while decimal != 0 and i < pre:
            result = int(decimal * 2)
            decimal = decimal * 2 - result
            decimal_convert = decimal_convert + str(result)
            i = i + 1
        else:
            decimal_convert += '0' * (pre - len(decimal_convert))
        # string_number 共34bits。正負號1bit，整數2bit，小數 'pre' bits
        string_integer2 = sign_b + ('00' + string_integer2)[-2:] + decimal_convert
    return string_integer2

def bTod(n, rawDataBit=18):
    '''
    把一個帶小數的二進制數n轉換成十進制
    小數點後面保留pre位小數
    '''
    pre = rawDataBit - 3

    sign_b, integer, n = -1 if int(n[0]) else 1, int(n[1:3]), '0.' + n[3:]

    string_number1 = str(n) #number1 表示二進制數，number2表示十進制數
    decimal = 0  #小數部分化成二進制後的值
    flag = False
    for i in string_number1: #判斷是否含小數部分
        if i == '.':
            flag = True
            break
    if flag: #若二進制數含有小數部分
        string_integer, string_decimal = string_number1.split('.') #分離整數部分和小數部分
        for i in range(len(string_decimal)):
            decimal += 2**(-i-1)*int(string_decimal[i])  #小數部分化成二進制
        number2 = int(str(int(string_integer, 2))) + decimal
        return (round(number2, pre) + integer) * sign_b
    else: #若二進制數只有整數部分
        return int(string_number1, 2)#若只有整數部分 直接一行代碼二進制轉十進制

def create_patchPairList(all_input, all_target):

    patchSize, margin = opt.patchSize, opt.margin
    patchPairList = []

    hPatchNum = round(all_input.shape[2]/patchSize)
    wPatchNum = round(all_input.shape[3]/patchSize)
    patchHeight = math.ceil(all_input.shape[2]/hPatchNum)
    patchWeight = math.ceil(all_input.shape[3]/wPatchNum)

    for r in range(hPatchNum):
        for c in range(wPatchNum):
            if c>0 and r>0:
                patchPair = [all_input[:,:,r*patchHeight-margin:(r+1)*patchHeight,c*patchWeight-margin:(c+1)*patchWeight],
                            all_target[:,:,r*patchHeight-margin:(r+1)*patchHeight, c*patchWeight-margin:(c+1)*patchWeight]]
            elif c>0:
                patchPair = [all_input[:,:,r*patchHeight:(r+1)*patchHeight,c*patchWeight-margin:(c+1)*patchWeight],
                            all_target[:,:,r*patchHeight:(r+1)*patchHeight, c*patchWeight-margin:(c+1)*patchWeight]]
            elif r>0:
                patchPair = [all_input[:,:,r*patchHeight-margin:(r+1)*patchHeight,c*patchWeight:(c+1)*patchWeight],
                            all_target[:,:,r*patchHeight-margin:(r+1)*patchHeight, c*patchWeight:(c+1)*patchWeight]]
            else:
                patchPair = [all_input[:,:,r*patchHeight:(r+1)*patchHeight,c*patchWeight:(c+1)*patchWeight],
                            all_target[:,:,r*patchHeight:(r+1)*patchHeight, c*patchWeight:(c+1)*patchWeight]]

            patchPairList.append(patchPair)

    return patchPairList, hPatchNum, wPatchNum

def nopatch(all_input, all_target):

    patchPairList = []
    hPatchNum = 1
    wPatchNum = 1
    patchPair = [all_input,all_target]
    patchPairList.append(patchPair)
    return patchPairList, hPatchNum, wPatchNum

def seed_torch():
    if opt.seed == 0:
        seed = random.randint(1, 10000)
    else:
        seed = opt.seed

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.backends.cudnn.benchmark = True 
    # 聽說開了可能會跑出不一樣的結果，但不開通常會慢
    torch.backends.cudnn.deterministic = True
    if not opt.no_cuda:
        torch.cuda.manual_seed(seed)

def compressImage(img, imageQF, codec='JPEG'):
    if codec == 'JPEG':
        retval, img_buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, imageQF])
        img_decoded = cv2.imdecode(img_buf, cv2.IMREAD_GRAYSCALE)
        encodedImageBitLength = len(img_buf)*8
    elif codec == 'JPEG2000':
        jp2 = glymur.Jp2k('myfile.jp2', data=img, cratios=[imageQF])
        jp2.layer = 0
        img_decoded = jp2[:]
        encodedImageBitLength = jp2.length * 8
    elif codec == 'WebP':
        retval, img_buf = cv2.imencode('.webp', img, [cv2.IMWRITE_WEBP_QUALITY, imageQF])
        img_decoded = cv2.imdecode(img_buf, cv2.IMREAD_GRAYSCALE)
        encodedImageBitLength = len(img_buf)*8
    elif codec == 'BPG':
        cv2.imwrite('temp.png', img)
        os.system('bpgenc -q {} temp.png -o temp.bpg'.format(imageQF)) # bpg encode
        os.system('bpgdec -o temp.png temp.bpg') # bpg decode
        img_decoded = cv2.imread('temp.png')[:,:,0]
        encodedImageBitLength = os.path.getsize('temp.bpg') * 8

    return img_decoded, encodedImageBitLength

class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self):
        return time.time() - self.t0

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0

class checkpoint():
    def __init__(self, opt):
        self.opt = opt
        self.ok = True
        self.dir = '../experiment/' + opt.save

        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        # Create folder if not exists
        def _make_dir(path):
            if not os.path.exists(path):
                os.makedirs(path)

        _make_dir(self.dir)
        _make_dir(self.dir + '/model')
        _make_dir(self.dir + '/results')

        # Check the log file exist or not, continue or create
        open_type = 'a' if os.path.exists(self.dir + '/log.txt') else 'w'
        self.log_file = open(self.dir + '/log.txt', open_type)
        with open(self.dir + '/config.txt', open_type) as f:
            f.write(now + '\n\n')
            for arg in vars(opt):
                f.write('{}: {}\n'.format(arg, getattr(opt, arg)))
            f.write('\n')

        ''' pre-defined codebook '''
        # load codebook
        self.codebookDict = {}
        with open('codebook/codebook_q{modelQF}.csv'.format(modelQF = opt.modelQF), 'r',newline='') as csvFile:
            rows = []
            for row in csv.reader(csvFile):
                rows.append(row)
            for i in range(2**opt.modelQF-1):
                self.codebookDict[int(rows[0][i])] = rows[1][i][2:]

        # make reverse_mapping (for decoding)
        self.reverse_mapping = {}
        for key, value in self.codebookDict.items():
            self.reverse_mapping[value] = key

    def load(self):
        my_model = model(self.opt).get_model()

        trainable = filter(lambda x: x.requires_grad and x.numel() != 4096, my_model.parameters())

        # Optimizer
        if self.opt.optimizer == 'SGD':
            optimizer_function = optim.SGD
            kwopt = {'momentum': self.opt.momentum}
        elif self.opt.optimizer == 'ADAM':
            optimizer_function = optim.Adam
            kwopt = {
                'betas': (self.opt.beta1, self.opt.beta2),
                'eps': self.opt.epsilon}
        elif self.opt.optimizernp.shape(im) == 'RMSprop':
            optimizer_function = optim.RMSprop
            kwopt = {'eps': self.opt.epsilon}

        kwopt['lr'] = self.opt.lr
        kwopt['weight_decay'] = self.opt.weight_decay
        my_optimizer = optimizer_function(trainable, **kwopt)

        # Adjust learning rate
        if self.opt.decay_type == 'step':
            my_scheduler = lrs.StepLR(
                my_optimizer,
                step_size=self.opt.lr_decay,
                gamma=self.opt.gamma)

        elif self.opt.decay_type.find('step') >= 0:
            milestones = self.opt.decay_type.split('_')
            milestones.pop(0)
            milestones = list(map(lambda x: int(x), milestones))
            my_scheduler = lrs.MultiStepLR(
                my_optimizer,
                milestones=milestones,
                gamma=self.opt.gamma)

        elif self.opt.decay_type == 'auto':
            my_scheduler = lrs.ReduceLROnPlateau(my_optimizer, mode='max', factor=0.7, threshold=0.001, patience=2)

        self.log_training = torch.Tensor()
        self.log_test = torch.Tensor()
        my_loss = loss(self.opt).get_loss()

        return my_model, my_loss, my_optimizer, my_scheduler

    def add_log(self, log, train=True):
        if train:
            self.log_training = torch.cat([self.log_training, log])
        else:
            self.log_test = torch.cat([self.log_test, log])

    def write_log(self, log, refresh=False):
        print(log)
        self.log_file.write(log + '\n')
        if refresh:
            self.log_file.close()
            self.log_file = open(self.dir + '/log.txt', 'a')

    def encode_quanWeight(self, trainer):
        encodedBits = ''
        header = (('0'*4 + bin(self.opt.modelQF))[2:])[-4:]  # header:儲存量化系數及每一層權重的量化區間
        ParamBits = ''                      # ParamBits:儲存量化後權重的編碼
        for layer, (name, module) in enumerate(trainer.model._modules.items()):
            if isinstance(module, nn.modules.Conv2d):
                header += dTob(module.interval, 20)
                paramData = np.array(module.quanWeight.view(module.quanWeight.numel()).cpu())
                for p in paramData:
                    ParamBits += self.codebookDict[p]
        encodedBits = header + ParamBits

        return encodedBits

    def done(self):
        self.log_file.close()


