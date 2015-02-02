import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import argparse

def loadData(XPath, yPath):
    X = np.genfromtxt(XPath, delimiter=",", dtype="float32")
    y = np.genfromtxt(yPath, delimiter=",", dtype="float32")
    return (X, y)

def standardize(featureVector):
    scaler = StandardScaler()
    return scaler.fit_transform(featureVector)

ap = argparse.ArgumentParser()
ap.add_argument("-x", "--xTrain", required = True,
    help = "path to training feature set")
ap.add_argument("-y", "--yTrain", required = True,
    help = "path to training target set")
ap.add_argument("-X", "--xTest", required = True,
    help = "path to testing feature set")
ap.add_argument("-Y", "--yTest", required = True,
    help = "path to testing target set")
args = vars(ap.parse_args())

(trainX, trainY) = loadData(args["xTrain"], args["yTrain"])
(testX, testY) = loadData(args["xTest"], args["yTest"])

# required scaling for SVM
trainX = standardize(trainX)
testX = standardize(testX)

labelCount = np.shape(trainY)[1] - 1

# init an array of SVRs, one for each label
SVR = [SVR() for _ in range(labelCount)]

for i,svr in enumerate(SVR):
    svr.fit(trainX, trainY[:,i+1])
    svr.predict(testX)