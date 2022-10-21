# basic and separable --- main_basic_and_separable

import random

import torch
import utils
from option import opt
from trainer import Trainer
import model.common as common

from importlib import import_module
import model
import numpy as np
import torch.nn as nn
import time
import csv
import copy
import os
import visdom

import math
import cv2
import PIL
import scipy.io as sio
from io import BytesIO
import matplotlib.pyplot as plt
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM


if __name__ == "__main__":
    currentTime = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    ST = time.time()
    try:
        os.makedirs('../experiment/' + opt.save)
    except FileExistsError:
        pass
    finally:
        # create csv log 
        logDict = ['QF', 'bpp', 'ori_psnr', 'out_psnr', 'out_ssim']
        with open(('../experiment/' + opt.save + '/' + 'encodedLog_Q{modelQF}_{}.csv').format(currentTime, modelQF = opt.modelQF), 'w', newline='') as csvFile:
            writer = csv.DictWriter(csvFile, logDict)
            writer.writeheader()

    bppDict = ['filename', 'bit','bpp','psnr','ssim','og-bit','og-bpp','og-psnr','og-ssim']
    with open('../experiment/' + opt.save + '/' + 'every_bpp_ch{}_patch{}_g{}.csv'.format(opt.ch,opt.patchSize,opt.n_clusters), 'w', newline='') as csvfile:
        writer2 = csv.writer(csvfile)
        writer2.writerow(bppDict)

    # set seed
    utils.seed_torch()

    # image list
    imageList = os.listdir(opt.data_path)
    imageList.sort()
    # number_of_Image = len(imageList)
    number_of_Image = [130]
    for imageQF in opt.imageQFs:
        singleQF_performanceLog = {'bpp':0, 'ori_psnr':0, 'out_psnr':0, 'out_ssim':0}
        # for idx_img in range(number_of_Image):
        for idx_img in (number_of_Image):
            # load image
            img = cv2.imread(opt.data_path + imageList[idx_img])[:,:,-1]
            print(imageList[idx_img])
            # create tensor (all_target, all_input, all_output)
            all_target = torch.Tensor(img.astype(float)).view(1,1,img.shape[0],img.shape[1])
            # produce encoded image
            img_decoded, encodedImageBitLength = utils.compressImage(img, imageQF, opt.codec)
            all_input = torch.tensor(img_decoded,dtype=torch.float32).view(all_target.shape)
            all_output = all_input.clone()

            if opt.pd_flag :
                all_input = utils.whole_image_padding(all_input)
                all_target = utils.whole_image_padding(all_target)
            # create patch pair list
            patchPairList, hPatchNum, wPatchNum = utils.create_patchPairList_v3(all_input, all_target)

            allpatch_glcm_feature = utils.glcm_feature(patchPairList,hPatchNum,wPatchNum)
            patch_group, label = utils.patch_grouping(patchPairList,allpatch_glcm_feature)
            
            total_encodedWeightBits = 0            
            for groupNum in range(len(patch_group)):

                # get current group
                cGroup = patch_group[groupNum]
                
                # label
                label_index = np.where(label == groupNum)[0].tolist()

                checkpoint = utils.checkpoint(opt)
                if checkpoint.ok:
                    t = Trainer(checkpoint, opt)
                    # t.inputData, t.target = t._prepare(cPatchPair[0], cPatchPair[1])
                    t.inputData, t.target = t._prepare_v2(cGroup[0],cGroup[1])

                    ''' run train '''
                    while not t.terminate():
                        t.train()
                        t.test()

                    ''' quantize weight '''
                    t.quantize_weight()

                    ''' encode_quanWeight '''
                    encodedWeightBits = checkpoint.encode_quanWeight(t)
                    total_encodedWeightBits += len(encodedWeightBits)

                    ''' encode_quanWeight '''
                    utils.rec_output(t.output,all_output,label_index,wPatchNum,hPatchNum)
                    # for index in range(len(label_index)):
                    #     patchNum = label_index[index]
                    #     r, c = patchNum//wPatchNum, patchNum%wPatchNum
                    #     patchHeight = math.ceil(all_input.shape[2]/hPatchNum)
                    #     patchWeight = math.ceil(all_input.shape[3]/wPatchNum)
                    #     # current patch size
                    #     cPatchHeight, cPatchWeight = all_output[0,0,r*patchHeight:(r+1)*patchHeight,c*patchWeight:(c+1)*patchWeight].shape
                    #     all_output[:,:,r*patchHeight:(r+1)*patchHeight,c*patchWeight:(c+1)*patchWeight] = t.output[index,:,-cPatchHeight:,-cPatchWeight:]

                    checkpoint.done()

            singleQF_performanceLog['bpp'] += (total_encodedWeightBits + encodedImageBitLength) / all_input.numel()
            singleQF_performanceLog['ori_psnr'] += utils.calc_PSNR(all_input, all_target, 0) 
            singleQF_performanceLog['out_psnr'] += utils.calc_PSNR(all_output, all_target, 0)
            singleQF_performanceLog['out_ssim'] += float(ssim(all_output.cuda(), all_target.cuda()))

            file_bit = total_encodedWeightBits + encodedImageBitLength
            file_bpp = file_bit/all_input.numel()
            file_psnr = utils.calc_PSNR(all_output, all_target, 0)
            file_ssim = float(ssim(all_output.cuda(), all_target.cuda()))
            og_bit = encodedImageBitLength
            og_bpp = encodedImageBitLength/all_input.numel()
            og_psnr = utils.calc_PSNR(all_input, all_target, 0)
            og_ssim = float(ssim(all_input.cuda(), all_target.cuda()))

            ''' save results '''
            if opt.save_results:
                filename = '{}/results/ours_{}_QF{}_{}_{}'.format(t.ckp.dir, opt.codec, imageQF, opt.dataset, imageList[idx_img])
                tmp = all_output.data[0].cpu()
                tmp = tmp.clamp(0,255).add(0.5).numpy().transpose(1,2,0).astype('uint8')[:, :, ::-1]
                cv2.imwrite('{}.png'.format(filename), tmp)
            with open('../experiment/' + opt.save + '/' + 'every_bpp_ch{}_patch{}_g{}.csv'.format(opt.ch,opt.patchSize,opt.n_clusters), 'a', newline='') as csvfile:
                writer2 = csv.writer(csvfile)
                file_name = 'ours_{}_QF{}_{}'.format(opt.codec, imageQF, imageList[idx_img])
                writer2.writerow([file_name, file_bit, '{:.4f}'.format(file_bpp),'{:.3f}'.format(file_psnr),'{:.5}'.format(file_ssim)
                                            ,og_bit,'{:.4f}'.format(og_bpp),'{:.3f}'.format(og_psnr),'{:.5}'.format(og_ssim)])


        for key in singleQF_performanceLog:
            singleQF_performanceLog[key] /= number_of_Image

        with open(('../experiment/' + opt.save + '/' + 'encodedLog_Q{modelQF}_{}.csv').format(currentTime, modelQF = opt.modelQF), 'a', newline='') as csvFile:
            writer = csv.DictWriter(csvFile, logDict)
            singleQF_performanceLog['QF'] = imageQF
            writer.writerow(singleQF_performanceLog) 

    endtime = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    ET = time.time()
    cost_time = ['time','{:.2f}s'.format(ET-ST)]
    with open(('../experiment/' + opt.save + '/' + 'encodedLog_Q{modelQF}_{}.csv').format(currentTime, modelQF = opt.modelQF), 'a', newline='') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(cost_time) 
    
    print('The endding time : ',endtime)


