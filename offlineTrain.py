from ORF import OnlineRandomForest
from ORFpy import dataRange
import numpy as np

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
