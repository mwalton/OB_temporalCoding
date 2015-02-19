import numpy as np
import matplotlib.pyplot as plt
from os import path
from matplotlib.pyplot import savefig

def loadData(XPath, yPath):
    X = np.genfromtxt(XPath, delimiter=",", dtype="float32")
    y = np.genfromtxt(yPath, delimiter=",", dtype="float32")
    return (X, y)

odorants = ['red', 'green', 'blue', 'orange']
BG=['grey','purple']
"""
xtrainpath="/Users/michaelwalton/Dropbox/Evolved Machines 2014/Machine Learning/datasets/BGtest/BG1/0.01train/sensorActivation.csv"
ytrainpath="/Users/michaelwalton/Dropbox/Evolved Machines 2014/Machine Learning/datasets/BGtest/BG1/0.01train/concentration.csv"
xtestpath="/Users/michaelwalton/Dropbox/Evolved Machines 2014/Machine Learning/datasets/BGtest/BG2/0.19test/sensorActivation.csv"
ytestpath="/Users/michaelwalton/Dropbox/Evolved Machines 2014/Machine Learning/datasets/BGtest/BG2/0.19test/concentration.csv"
"""
xtrainpath="/Users/michaelwalton/Dropbox/Evolved Machines 2014/Machine Learning/datasets/kaggle/paul_medC_BG2/train/sensorActivation.csv"
ytrainpath="/Users/michaelwalton/Dropbox/Evolved Machines 2014/Machine Learning/datasets/kaggle/paul_medC_BG2/train/concentration.csv"
xtestpath="/Users/michaelwalton/Dropbox/Evolved Machines 2014/Machine Learning/datasets/kaggle/paul_highC_BG1/test/sensorActivation.csv"
ytestpath="/Users/michaelwalton/Dropbox/Evolved Machines 2014/Machine Learning/datasets/kaggle/paul_highC_BG1/test/concentration.csv"

redPath="/Users/michaelwalton/Dropbox/Evolved Machines 2014/Machine Learning/datasets/kaggle/paul_medC_BG2/r"
greenPath="/Users/michaelwalton/Dropbox/Evolved Machines 2014/Machine Learning/datasets/kaggle/paul_medC_BG2/g"
bluePath="/Users/michaelwalton/Dropbox/Evolved Machines 2014/Machine Learning/datasets/kaggle/paul_medC_BG2/b"
yellowPath="/Users/michaelwalton/Dropbox/Evolved Machines 2014/Machine Learning/datasets/kaggle/paul_medC_BG2/y"

rX=path.join(redPath,   "sensorActivation.csv")
rY=path.join(redPath,   "concentration.csv")
gX=path.join(greenPath, "sensorActivation.csv")  
gY=path.join(greenPath, "concentration.csv")     
bX=path.join(bluePath,  "sensorActivation.csv")  
bY=path.join(bluePath,  "concentration.csv")     
yX=path.join(yellowPath,"sensorActivation.csv")  
yY=path.join(yellowPath,"concentration.csv")     
"""
#/Users/michaelwalton/Dropbox/Evolved Machines 2014/Machine Learning/datasets/kaggle/MotifsAff
basePath="/Users/michaelwalton/Dropbox/Evolved Machines 2014/Machine Learning/datasets/NICE_viz"

xtrainpath=path.join(basePath, "m5a5train/sensorActivation.csv")
ytrainpath=path.join(basePath, "m5a5train/concentration.csv")
xtestpath=path.join(basePath, "m5a5test/sensorActivation.csv")
ytestpath=path.join(basePath, "m5a5test/concentration.csv")
"""
# odorant label ranges
#oRange = {'red' : [0,732], 'green' : [0,732], 'blue' : [0,732], 'yellow' : [0,732]}

#(Xtrain, ytrain) = loadData(xtrainpath, ytrainpath)

training_set={'red' : loadData(rX, rY),
              'green' : loadData(gX, gY),
              'blue' : loadData(bX, bY),
              'orange' : loadData(yX, yY) }

(Xtest, ytest) = loadData(xtestpath, ytestpath)

tTest=np.shape(ytest)[0]
yMax=np.max(ytest)
for o in odorants:
    oX,oY=training_set[o]
    if np.max(oY > yMax):
        yMax=np.max(oY)

for i, o in enumerate(odorants):
    oX,oY = training_set[o]
    oX=oX[0:tTest,:]
    oY=oY[0:tTest,:]
    
    fig = plt.figure(figsize=(10,10))
    """
    ax1 = fig.add_subplot(311)
    ax1.plot(ytrain[oRange[o][0]:oRange[o][1],i+1], c=o)
    #ax1.set_xlabel('Time')
    ax1.set_ylabel('log[c]')
    ax1.set_yscale('log')
    ax1.set_ylim(1e-3, np.max(ytrain))
    """
    ax1 = fig.add_subplot(211)
    ax1.plot(oY[:,0], c=BG[0])
    ax1.plot(oY[:,i+1], c=o)
    ax1.set_ylabel('[%s]' % o)
    ax1.set_ylim(0, yMax)
    ax1.set_xlim(0, np.shape(oY)[0])
    
    ax2 = fig.add_subplot(212)
    ax2.imshow(np.transpose(oX), aspect='auto', interpolation='nearest')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Sensor Activation')
    
    #plt.show()
    savefig(path.join('plots',o+'.png'))
    
fig = plt.figure(figsize=(10,10))
"""    
ax1 = fig.add_subplot(211)
ax1.plot(ytest[:,1], c=odorants[0])
ax1.plot(ytest[:,2], c=odorants[1])
ax1.plot(ytest[:,3], c=odorants[2])
ax1.plot(ytest[:,4], c=odorants[3])
#ax1.set_xlabel('Time')
ax1.set_ylabel('log[c]')
ax1.set_yscale('log')
ax1.set_ylim(1e-2, np.max(ytest))
"""
ax2 = fig.add_subplot(211)
ax2.plot(ytest[:,0], c=BG[1])
ax2.plot(ytest[:,1], c=odorants[0])
ax2.plot(ytest[:,2], c=odorants[1])
ax2.plot(ytest[:,3], c=odorants[2])
ax2.plot(ytest[:,4], c=odorants[3])
ax2.set_ylabel('[c]')
ax2.set_ylim(0, yMax)
ax2.set_xlim(0, np.shape(ytest)[0])

ax2 = fig.add_subplot(212)
ax2.imshow(np.transpose(Xtest), aspect='auto', interpolation='nearest')
ax2.set_xlabel('Time')
ax2.set_ylabel('Sensor Activation')

savefig(path.join('plots', 'mix.png'))

#plt.show()