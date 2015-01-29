import matplotlib.pyplot as plt
#from Image import NEAREST
#from matplotlib.cm import cmap_d
import argparse
import os.path
import numpy as np
from casuarius import required
from sklearn.metrics import accuracy_score
from os import listdir
#import pylab as pl

def evaluateNS(path):
    d = np.genfromtxt(path, delimiter=",", dtype="float32", skip_header=1)
    
    ctx_pred = d[:,1]
    f_pred = d[:,0]
    target = d[:,2]
    
    return (accuracy_score(target, f_pred), accuracy_score(target, ctx_pred))

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
ap.add_argument("-S", "--sameNS", required=True,
    help = "path to the NS data you want to plot for same condition")
ap.add_argument("-D", "--diffNS", required=True,
    help = "path to the NS data you want to plot for the diff condition")
args = vars(ap.parse_args())
  
(sH, sX, sY) = loadData(args["same"])
(dH, dX, dY) = loadData(args["diff"])

trainResult = np.genfromtxt(args["trainResult"], delimiter=",", dtype=float, skip_header=1)
mean_BGc = trainResult[0]


######################## EVALUATE NS
sFolders = listdir(args["sameNS"])
dFolders = listdir(args["diffNS"])

ns_sY = []
ns_dY = []

ctx_sY = []
ctx_dY = []
for sF in sFolders:
    if not sF.startswith('.'):
        (tempFib, tempCtx) = evaluateNS(os.path.join(args["sameNS"], sF, "tstepAccuracy.csv"))
        ns_sY.append(tempFib)
        ctx_sY.append(tempCtx)

for dF in dFolders:
    if not dF.startswith('.'):
        (tempFib, tempCtx) = evaluateNS(os.path.join(args["diffNS"], dF, "tstepAccuracy.csv"))
        ns_dY.append(tempFib)
        ctx_dY.append(tempCtx)
"""
ns_sY = np.array(ns_sY)
ns_dY = np.array(ns_dY)

np.reshape(ns_sY, np.shape(sX))
np.reshape(ns_dY, np.shape(dX))
"""
######################## PLOT
fig = plt.figure()
ax1 = fig.add_subplot(111)
#ax1.scatter(sX, sY[:,0],c='b', marker='s', label='same BG')
samePlt = ax1.plot(sX, sY[:,0], '-s', c='b', label='RBM same BG')
diffPlt = ax1.plot(dX, dY[:,0], '-o', c='r', label='RBM diff BG')

sameNS_plt = ax1.plot(sX, ns_sY, '-s', c='g', label='Fibers same BG')
diffNS_plt = ax1.plot(dX, ns_dY, '-o', c='y', label='Fibers diff BG')
sameCtx_plt = ax1.plot(sX, ctx_sY, '-s', c='orange', label='Ctx same BG')
diffCtx_plt = ax1.plot(dX, ctx_dY, '-o', c='purple', label='Ctx diff BG')

ax1.set_title(sH[2])
box = ax1.get_position()
ax1.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])
ax1.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, -0.05),
           handles=[samePlt, diffPlt, sameNS_plt, diffNS_plt, sameCtx_plt, diffCtx_plt],
           labels=['RBM same BG', 'RBM diff BG', 'NS same BG', 'NS diff BG', 'Ctx same BG', 'Ctx diff BG'])
ax1.axvline(mean_BGc, color='y')
ax1.set_xlabel('Mean Background Concentration')
ax1.set_ylabel('Accuracy')
#ax1.set_ylim([0.85, 1.0])
#ax1.scatter(dX, dY[:,0],c='r', marker='o', label='diff BG')

if (not args["fileOut"] == ""):
    plt.savefig(args["fileOut"])
else:
    plt.show()

#print(same)
