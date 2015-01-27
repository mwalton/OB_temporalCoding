import matplotlib.pyplot as plt
#from Image import NEAREST
#from matplotlib.cm import cmap_d
import argparse

import numpy as np
#import pylab as pl

def loadData(path):
    d = np.genfromtxt(path, delimiter=",", dtype=None)
    
    #header is the first row
    h = d[0,:]
    
    #select all rows in d where classifier == RBM
    i = d[:,1] == 'rbm'
    d = d[i]
    #first col = BGc = x values of scatter plot
    dX = (d[:, 0]).astype(np.float)
    #remaining columns are accuracy values
    dY = (d[:, range(2,np.shape(d)[1])]).astype(np.float)
    
    #return tuple containing feature and target vectors
    return (h, dX, dY)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--same", required = True,
    help = "path to same train / test BG results")
ap.add_argument("-d", "--diff", required = True,
    help = "path to diff train / test BG results")
ap.add_argument("-t", "--trainResult", required = True,
    help = "training results output")
ap.add_argument("-f", "--fileOut", default="",
    help = "if defined, plot will be written to file instead of displayed")
args = vars(ap.parse_args())
  
(sH, sX, sY) = loadData(args["same"])
(dH, dX, dY) = loadData(args["diff"])

trainResult = np.genfromtxt(args["trainResult"], delimiter=",", dtype=float, skip_header=1)
mean_BGc = trainResult[0]

fig = plt.figure()
ax1 = fig.add_subplot(111)
#ax1.scatter(sX, sY[:,0],c='b', marker='s', label='same BG')
samePlt = ax1.plot(sX, sY[:,0], '-o', c='b', label='same BG')
diffPlt = ax1.plot(dX, dY[:,0], '-o', c='r', label='diff BG')
ax1.set_title(sH[2])
ax1.legend(handles=[samePlt, diffPlt], labels=['same BG', 'diff BG'])
ax1.axvline(mean_BGc, color='y')
ax1.set_xlabel('Mean Background Concentration')
ax1.set_ylabel('Accuracy')
ax1.set_ylim([0.85, 1.0])
#ax1.scatter(dX, dY[:,0],c='r', marker='o', label='diff BG')

if (not args["fileOut"] == ""):
    plt.savefig(args["fileOut"])
else:
    plt.show()

#print(same)
