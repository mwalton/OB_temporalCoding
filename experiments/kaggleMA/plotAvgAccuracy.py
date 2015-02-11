import matplotlib.pyplot as plt
#from Image import NEAREST
#from matplotlib.cm import cmap_d
import argparse
from os import listdir
from os import path

import numpy as np
#import pylab as pl

def getIndependentVarIndicies(files, delimiter):
    #here we just want to be able to get all the file prefixes and return a sorted
    #array containing exactly one occurance of each prefix
    n = []
    for f in files:
        if delimiter in f:
            n.append(f.split(delimiter)[0])
        
    i = np.array(n)
    i = np.unique(i)
    return i

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
ap.add_argument("-r", "--recursive", required = True,
    help = "folder containing datasets to be averaged")
ap.add_argument("-s", "--same", required = True,
    help = "path to same train / test BG results")
ap.add_argument("-d", "--diff", required = True,
    help = "path to diff train / test BG results")
ap.add_argument("-t", "--trainResult", required = True,
    help = "training results output")
ap.add_argument("-f", "--fileOut", default="",
    help = "if defined, plot will be written to file instead of displayed")
ap.add_argument("-l", "--label", default="",
    help = "the prefix datalabel (usually maxBGc)")
args = vars(ap.parse_args())

files = listdir(args["recursive"])
indVar = getIndependentVarIndicies(files, "diff")

train_mean_BGc = []
diff_averages = []
same_averages = []

for i in indVar:
    tr = path.join(args["recursive"], i + args["trainResult"])
    trainResult = np.genfromtxt(tr, delimiter=",", dtype=float, skip_header=1)
    train_mean_BGc.append(trainResult[0].astype(np.float))
    
    sPath = path.join(args["recursive"], i + args["same"])
    dPath = path.join(args["recursive"], i + args["diff"])
    (sH, sX, sY) = loadData(sPath)
    (dH, dX, dY) = loadData(dPath)
    
    same_averages.append(np.mean(sY[:,0]))
    diff_averages.append(np.mean(dY[:,0]))

# fit the data using least squares polyfit
same_coeff = np.polyfit(train_mean_BGc, same_averages, 3)
same_poly = np.poly1d(same_coeff)
_same_averages = same_poly(train_mean_BGc)

diff_coeff = np.polyfit(train_mean_BGc, diff_averages, 3)
diff_poly = np.poly1d(diff_coeff)
_diff_averages = diff_poly(train_mean_BGc)
    
#plot
fig = plt.figure()
ax1 = fig.add_subplot(111)
samePlt = ax1.scatter(train_mean_BGc, same_averages, c='green', marker='s', label='same BG')
diffPlt = ax1.scatter(train_mean_BGc, diff_averages, c='orange', marker='o', label='diff BG')
ax1.plot(train_mean_BGc, _same_averages, c='green')
ax1.plot(train_mean_BGc, _diff_averages, c='orange')

#samePlt = ax1.plot(train_mean_BGc, same_averages, '-o', c='g', label='same BG')
#diffPlt = ax1.plot(train_mean_BGc, diff_averages, '-o', c='orange', label='diff BG')
ax1.set_title("Average Accuracy Score")
ax1.legend(handles=[samePlt, diffPlt], labels=['same BG', 'diff BG'])
ax1.set_xlabel('Mean Training Background Concentration')
ax1.set_ylabel('Average Testing Accuracy for all BGc')
ax1.set_ylim([0.85, 1.0])
ax1.set_xlim([0.0, train_mean_BGc[len(train_mean_BGc) - 1]])

if (not args["fileOut"] == ""):
    plt.savefig(args["fileOut"])
else:
    plt.show()
