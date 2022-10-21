import math
import random
import numpy as np
import matplotlib.pyplot as plt
class Canopy:
    def __init__(self, dataset):
        self.dataset = dataset
        self.t1 = 7
        self.t2 = 5
    # 设置初始阈值
    def setThreshold(self, t1, t2):
        if t1 > t2:
            self.t1 = t1
            self.t2 = t2
        else:
            print('t1 needs to be larger than t2!')
    # 使用欧式距离进行距离的计算
    def euclideanDistance(self, vec1, vec2):
        return math.sqrt(((vec1 - vec2)**2).sum())
    # 根据当前dataset的长度随机选择一个下标
    def getRandIndex(self):
        return random.randint(0, len(self.dataset) - 1)
    def dis_info(self):
        if self.t1 == 0:
            print('Please set the threshold.')
        else:
            canopies = []  # 用于存放最终归类结果
            dislist = []
            while len(self.dataset) != 0:
                rand_index = self.getRandIndex()
                current_center = self.dataset[rand_index]  # 随机获取一个中心点，定为P点
                current_center_list = []  # 初始化P点的canopy类容器
                delete_list = []  # 初始化P点的删除容器
                self.dataset = np.delete(
                    self.dataset, rand_index, 0)  # 删除随机选择的中心点P
                for datum_j in range(len(self.dataset)):
                    datum = self.dataset[datum_j]
                    distance = self.euclideanDistance(
                        current_center, datum)  # 计算选取的中心点P到每个点之间的距离
                    dislist.append(distance)
        return np.max(np.array(dislist)),np.min(np.array(dislist)),np.std(np.array(dislist))
    def clustering(self):
        if self.t1 == 0:
            print('Please set the threshold.')
        else:
            canopies = []  # 用于存放最终归类结果
            while len(self.dataset) != 0:
                rand_index = self.getRandIndex()
                current_center = self.dataset[rand_index]  # 随机获取一个中心点，定为P点
                current_center_list = []  # 初始化P点的canopy类容器
                delete_list = []  # 初始化P点的删除容器
                self.dataset = np.delete(
                    self.dataset, rand_index, 0)  # 删除随机选择的中心点P
                for datum_j in range(len(self.dataset)):
                    datum = self.dataset[datum_j]
                    distance = self.euclideanDistance(
                        current_center, datum)  # 计算选取的中心点P到每个点之间的距离
                    if distance < self.t1:
                        # 若距离小于t1，则将点归入P点的canopy类
                        current_center_list.append(datum)
                    if distance < self.t2:
                        delete_list.append(datum_j)  # 若小于t2则归入删除容器
                # 根据删除容器的下标，将元素从数据集中删除
                self.dataset = np.delete(self.dataset, delete_list, 0)
                canopies.append((current_center, current_center_list))
        return canopies