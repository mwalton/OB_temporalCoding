import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
from sklearn.preprocessing import StandardScaler

from sklearn import decomposition
from sklearn import datasets

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--file", required = True,
    help = "path to the evolution run file")
ap.add_argument("-l", "--label", required=True,
    help = "label file")
ap.add_argument("-u", "--x0", required=True, type=int,
    help = "first component")
ap.add_argument("-v", "--x1", required=True, type=int,
    help = "second component")
ap.add_argument("-w", "--x2", required=True, type=int,
    help = "third component")
ap.add_argument("-Y", "--Y", required=True, type=int,
    help = "accuracy")
args = vars(ap.parse_args())

raw = np.genfromtxt(args["file"], delimiter=',', dtype=float)
labels = np.genfromtxt(args["label"], delimiter=',', dtype=None)

xIdx = [args["x0"], args["x1"], args["x2"]]

n = np.shape(raw)[0]
X = raw[:,xIdx]
y = raw[:,args["Y"]]

fig = plt.figure(1, figsize=(8, 6))
plt.clf()
ax = Axes3D(fig, elev=48, azim=134)

plt.cla()

ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.spectral)
ax.set_title(labels[args["Y"]] + " n=" + str(n))
ax.set_xlabel(labels[xIdx[0]])
ax.set_ylabel(labels[xIdx[1]])
ax.set_zlabel(labels[xIdx[2]])

plt.show()