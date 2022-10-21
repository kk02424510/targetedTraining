import torch
from option import opt
from model import model
import torch.nn as nn
import os
import numpy as np
import csv
from tqdm import tqdm
import statistics
import matplotlib.pyplot as plt
import scipy.stats as st
import huffman
import math
from sklearn import cluster
import pickle
from libKMCUDA import kmeans_cuda

load_path = './K_codebook/layer_para'
# data = []
save_flag = True
c_times = False
show = False
make_codebook = False
dirs = os.listdir(load_path)
para_layer_list = []

n_clusters = (1<<opt.modelQF-1)*2-1
for i in dirs:
    if os.path.splitext(i)[1] == '.txt':
        para_layer_list.append(i)
num_of_file = len(para_layer_list)
# def quan(data):
#     bit = (1<<opt.modelQF-1)
#     table_dict = {v:0 for v in list(range(-bit,bit,1))}
#     scale = 

for num in range(num_of_file):
    data = []
    flie_name = para_layer_list[num].split('.')[0]
    print('current layer:{}'.format(flie_name))
    # print('save name:{}'.format(para_layer_list[num].split('.')[0]))
    with open(load_path +'/'+ para_layer_list[num], 'r', encoding='utf8') as f:
        for i in f:
            data.append([float(j) for j in i.split()])

    data_np = np.array(data[0])
    x = data_np.reshape((-1,1))
    centroids, assignments = kmeans_cuda(x, 4, verbosity=1, seed=3)
    # k_means = cluster.KMeans(n_clusters=n_clusters)
    # k_means.fit(x)
    # values = k_means.cluster_centers_.squeeze()
    # labels = k_means.labels_
    # values = np.sort(values)
    
    # OG_c_min = np.percentile(data_np,0.1)
    # OG_c_max = np.percentile(data_np,99.9)
    print('OG data len = {}'.format(len(data_np)))
    print('OG parameter max = {}'.format(np.max(data_np)))
    print('OG parameter min = {}'.format(np.min(data_np)))
    print('OG parameter mean = {}'.format(np.mean(data_np)))
    print('OG parameter std = {}'.format(np.std(data_np)))
    # print('OG para 1 / 99 c: {} , {}'.format(OG_c_min,OG_c_max))

    c_min = np.percentile(data_np,0.1)
    c_max = np.percentile(data_np,99.9)
    d_clip = data_np[np.logical_and(data_np>=c_min, data_np<=c_max)]

    d_max = np.max(d_clip)
    d_min = np.min(d_clip)
    d_mean = np.mean(d_clip)
    d_std = np.std(d_clip)
    # ci_min, ci_max = st.norm.interval(0.05, loc = d_mean, scale=st.sem(data_np)

    print('cliped data len = {}'.format(len(d_clip)))
    print('clip data = {}, {}%'.format(len(data_np)-len(d_clip),(len(data_np)-len(d_clip))/len(data_np)))
    print('parameter max = {}'.format(d_max))
    print('parameter min = {}'.format(d_min))
    print('parameter mean = {}'.format(d_mean))
    print('parameter std = {}'.format(d_std))
    print('para 1 / 99 c: {} , {}'.format(c_min,c_max))


    if c_times:
        bit = (1<<opt.modelQF-1)
        table_dict = {v:0 for v in list(range(-bit+1,bit,1))}
        scale = (round(np.max(data_np))-round(np.min(data_np)))/((bit-1)*2)
        print(scale)
        # zp = round((bit-1)-(np.max(data_np))/scale)
        zp = 0
        for i in tqdm(data_np):
            Q = round(i/scale + zp)
            table_dict[Q] += 1

        # times_para = list(table_dict.values())

    if make_codebook:
        input = [(k,v) for k,v in table_dict.items()]
        # input = [(str(int(data[0][i])),int(data[1][i])) for i in range(len(data[0][:]))]
        codebook = huffman.codebook(input)

    if save_flag:
        try:
            os.makedirs(load_path)
        except FileExistsError:
            pass
        finally:
            pickle.dump(k_means, open('codebook_Q{modelQF}_{filename}_Kmean.pkl'.format( modelQF = opt.modelQF, filename = flie_name), 'wb'))
            # with open((load_path +'/'+ 'codebook_Q{modelQF}_{filename}_Kmean.csv').format( modelQF = opt.modelQF, filename = flie_name), 'w', newline='') as csvFile:
            #     # writer = csv.DictWriter(csvFile, codebook)
            #     writer = csv.writer(csvFile)
            #     # writer.writeheader()
            #     # writer2 = csv.writer(csvFile)
            #     writer.writerow(values)
    # data_hist = np.histogram(data_np)
    # c ,bins = np.histogram(data_np)
    # bins = list(np.arange(d_min, d_max, 0.1))
    if show:
        plt.hist(x, bins=256 ,color='.5', edgecolor='.5')
        # plt.hist(d_clip, bins=256 )
        plt.title("histogram {}".format(para_layer_list[num]))
        values = np.sort(values)
        # plt.text(2,400000, 'min={}, max={}'.format(d_max,d_min))
        for center_1, center_2 in zip(values[:-1], values[1:]):
            plt.axvline(.5 * (center_1 + center_2), color='b')
        # for center_1, center_2 in zip(regular_values[:-1], regular_values[1:]):
        #     plt.axvline(.5 * (center_1 + center_2), color='b', linestyle='--')
        plt.show()
    f.close()
    print('Finish!')
