import numpy as np
import matplotlib.pyplot as plt
from os import path
from matplotlib.pyplot import savefig

def loadData(XPath, yPath):
    X = np.genfromtxt(XPath, delimiter=",", dtype="float32")
    y = np.genfromtxt(yPath, delimiter=",", dtype="float32")
    return (X, y)

odorants = ['red', 'green', 'blue', 'yellow']
"""
xtrainpath="/Users/michaelwalton/Dropbox/Evolved Machines 2014/Machine Learning/datasets/BGtest/BG1/0.01train/sensorActivation.csv"
ytrainpath="/Users/michaelwalton/Dropbox/Evolved Machines 2014/Machine Learning/datasets/BGtest/BG1/0.01train/concentration.csv"
xtestpath="/Users/michaelwalton/Dropbox/Evolved Machines 2014/Machine Learning/datasets/BGtest/BG2/0.19test/sensorActivation.csv"
ytestpath="/Users/michaelwalton/Dropbox/Evolved Machines 2014/Machine Learning/datasets/BGtest/BG2/0.19test/concentration.csv"
"""
#/Users/michaelwalton/Dropbox/Evolved Machines 2014/Machine Learning/datasets/kaggle/MotifsAff
basePath="/Users/michaelwalton/Dropbox/Evolved Machines 2014/Machine Learning/datasets/NICE_viz"

xtrainpath=path.join(basePath, "m5a5train/sensorActivation.csv")
ytrainpath=path.join(basePath, "m5a5train/concentration.csv")
xtestpath=path.join(basePath, "m5a5test/sensorActivation.csv")
ytestpath=path.join(basePath, "m5a5test/concentration.csv")

# odorant label ranges
oRange = {'red' : [0,399], 'green' : [400,799], 'blue' : [800,1199], 'yellow' : [1200,1599]}

(Xtrain, ytrain) = loadData(xtrainpath, ytrainpath)
(Xtest, ytest) = loadData(xtestpath, ytestpath)

for i, o in enumerate(odorants):
    fig = plt.figure(figsize=(10,10))
    
    ax1 = fig.add_subplot(311)
    ax1.plot(ytrain[oRange[o][0]:oRange[o][1],i+1], c=o)
    #ax1.set_xlabel('Time')
    ax1.set_ylabel('log[c]')
    ax1.set_yscale('log')
    ax1.set_ylim(1e-3, np.max(ytrain))
    
    ax2 = fig.add_subplot(312)
    ax2.plot(ytrain[oRange[o][0]:oRange[o][1],i+1], c=o)
    #ax2.set_xlabel('Time')
    ax2.set_ylabel('[c]')
    ax2.set_ylim(1e-3, np.max(ytrain))
    
    ax3 = fig.add_subplot(313)
    ax3.imshow(np.transpose(Xtrain)[:,oRange[o][0]:oRange[o][1]], aspect='auto', interpolation='nearest')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Sensor Activation')
    
    #plt.show()
    savefig(path.join('plots',o+'.png'))
    
fig = plt.figure(figsize=(10,10))
    
ax1 = fig.add_subplot(311)
ax1.plot(ytest[:,1], c=odorants[0])
ax1.plot(ytest[:,2], c=odorants[1])
ax1.plot(ytest[:,3], c=odorants[2])
ax1.plot(ytest[:,4], c=odorants[3])
#ax1.set_xlabel('Time')
ax1.set_ylabel('log[c]')
ax1.set_yscale('log')
ax1.set_ylim(1e-2, np.max(ytest))

ax2 = fig.add_subplot(312)
ax2.plot(ytest[:,1], c=odorants[0])
ax2.plot(ytest[:,2], c=odorants[1])
ax2.plot(ytest[:,3], c=odorants[2])
ax2.plot(ytest[:,4], c=odorants[3])
#ax2.set_xlabel('Time')
ax2.set_ylabel('[c]')
ax2.set_ylim(0, np.max(ytest))

ax3 = fig.add_subplot(313)
ax3.imshow(np.transpose(Xtest), aspect='auto', interpolation='nearest')
ax3.set_xlabel('Time')
ax3.set_ylabel('Sensor Activation')

savefig(path.join('plots', 'mix.png'))

#plt.show()