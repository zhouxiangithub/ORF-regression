import numpy as np
import math
from ORFpy import ORF, dataRange

def g(x):
    return math.sin(x[0]) if x[0]<x[1] else math.cos(x[1]+math.pi/2)

n = 10
X = np.random.randn(n,2)
y = map(g,X)

param = {'minSamples': 10, 'minGain': 0, 'xrng': dataRange(X), 'maxDepth': 10}
xtest = np.random.randn(n,2)
ytest = map(g,xtest)
orf = ORF(param,numTrees=50)
for i in range(n):
    orf.update(X[i,:],y[i])

preds = orf.predicts(xtest)
sse = sum( map(lambda z: (z[0]-z[1])*(z[0]-z[1]) , zip(preds,ytest)) )
rmse = math.sqrt(sse / float(len(preds)))
print "RMSE: " + str(round(rmse,2))
# RMSE: 0.22
