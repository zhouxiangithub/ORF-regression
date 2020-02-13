#!/usr/bin/python
# -*- coding:UTF-8 -*-

import numpy as np
import math
from ORFpy import ORF, dataRange
from sklearn.externals import joblib
import os

class OnlineRandomForest:
    # 初始化，神经网络参数，树的数量，数据集路径，训练集与测试集的比例
    def __init__(self, param, numTrees,input,dataPath='./data/offline/',pct=0.9):
        self.input = input
        self.param = param
        self.pct = pct
        self.data = self.get_data(dataPath)
        self.init_data()
        # self.param['xrng'] = dataRange(self.trainX)
        self.orf = ORF(param, numTrees=numTrees)

    def get_data(self,dataPath):
        # 训练的文件全部在data文件夹下，datapath为data的路径，读取该文件夹下的数据文件
        filePath = []
        pathDir = os.listdir(dataPath)
        for allDir in pathDir:
            child = os.path.join('%s%s' % (dataPath, allDir))
            filePath.append(child)
        data = []
        for path in filePath:
            fopen = open(path, 'r')
            for eachLine in fopen:
                eachLineData = eachLine.split(",")
                eachLineData = np.array(map(lambda x:float(x),eachLineData))
                data.append(eachLineData)
            fopen.close()
        return np.array(data)

    # 按照比例分隔数据集，构造训练集和测试集
    def init_data(self):
        self.data = self.data[0:100]
        self.trainX = self.data[:int(len(self.data)*self.pct),:self.input]
        self.trainY = self.data[:int(len(self.data)*self.pct),self.input:self.input+1]
        self.testX = self.data[int(len(self.data)*self.pct):,:self.input]
        self.testY = self.data[int(len(self.data)*self.pct):,self.input:self.input+1]
        print self.trainX.shape
        print self.trainY.shape

    # 离线训练: EPOCH代表迭代次数（暂时不知道迭代次数是否起作用，默认为1）
    def offline_train(self,EPOCH=1):
        print "train start..."
        for epoch in range(EPOCH):
            for i in range(len(self.trainY)):
                self.orf.update(self.trainX[i, :], self.trainY[i])
            print epoch
            self.test()
        print "train finish!"
        self.save()
        print "save model!"


    # 在线训练: newData代表新产生的data,EPOCH代表迭代次数（暂时不知道迭代次数是否起作用，默认为1），flag为true代表保留旧数据，false代表替换旧数据
    def online_train(self,newDataPath="./data/online/",EPOCH=1,flag=False):
        print "load model..."
        self.load()
        print "Online train start..."
        newData = self.get_data(newDataPath)
        if flag:
            self.data = np.concatenate([self.data,newData],0)
        else:
            self.data = newData
        self.init_data()
        # for ort in self.orf.forest:
        #     ort.tree.elem.xrng = dataRange(self.trainX)
        self.offline_train(EPOCH=5)
        print "Online train finish!"

    def save(self):
        # 保存Model(注:save文件夹要预先建立，否则会报错)
        joblib.dump(self.orf, "./save/rf.pkl")

    def load(self):
        # 读取Model
        self.orf = joblib.load("./save/rf.pkl")

    def test(self):
        preds = self.orf.predicts(self.testX)
        # RMSE 均方根误差亦称标准误差
        sse = sum(map(lambda z: (z[0] - z[1]) * (z[0] - z[1]), zip(preds, self.testY)))
        rmse = math.sqrt(sse / float(len(preds)))
        print "RMSE: " + str(round(rmse, 2)) + "\n"

if __name__ == "__main__":
    fopen = open("data/data.csv", 'r')
    tempData = []
    for eachLine in fopen:
        eachLineData = eachLine.split(",")
        eachLineData = np.array(map(lambda x: float(x), eachLineData))
        tempData.append(eachLineData)
    tempData = np.array(tempData)
    fopen.close()
    param = {'minSamples': 10, 'minGain': 0, 'xrng': dataRange(tempData[:,:722]), 'maxDepth': 10}
    orf = OnlineRandomForest(param=param,numTrees=50,input=722)
    orf.offline_train()
    orf.online_train()
