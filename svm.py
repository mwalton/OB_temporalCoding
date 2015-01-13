import numpy as np
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV
import time
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

def loadData(XPath, yPath):
    X = np.genfromtxt(XPath, delimiter=",", dtype="float32")
    y = np.genfromtxt(yPath, delimiter=",", dtype="float32")
    return (X, y)
    
def convertToClasses(targetVector):
    return np.argmax(targetVector[:,1:5], axis=1)

def standardize(featureVector):
    scaler = StandardScaler()
    return scaler.fit_transform(featureVector)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-x", "--xTrain", required = True,
    help = "path to training feature set")
ap.add_argument("-y", "--yTrain", required = True,
    help = "path to training target set")
ap.add_argument("-X", "--xTest", required = True,
    help = "path to testing feature set")
ap.add_argument("-Y", "--yTest", required = True,
    help = "path to testing target set")
ap.add_argument("-o", "--optomize", type = int, default = 0,
    help = "whether or not a grid search should be performed")
ap.add_argument("-m", "--multiClass", type = int, default=1,
    help = "exclusive multi class or regression")
args = vars(ap.parse_args())

(trainX, trainY) = loadData(args["xTrain"], args["yTrain"])
(testX, testY) = loadData(args["xTest"], args["yTest"])

# required scaling for SVM
trainX = standardize(trainX)
testX = standardize(testX)

if (args["multiClass"] == 1):
    trainY = convertToClasses(trainY)
    testY = convertToClasses(testY)
    
# check to see if a grid search should be done
if args["optomize"] == 1:
    # perform a grid search on the 'C' and 'gamma' parameter
    # of SVM
    print "SEARCHING SVM"
    params = {"C": [1.0, 10.0, 100.0]}
    start = time.time()
    gs = GridSearchCV(svm.SVC(), params, n_jobs = -1, verbose = 1)
    gs.fit(trainX, trainY)
 
    # print diagnostic information to the user and grab the
    # best model
    print "done in %0.3fs" % (time.time() - start)
    print "best score: %0.3f" % (gs.best_score_)
    print "SVM PARAMETERS"
    bestParams = gs.best_estimator_.get_params()
 
    # loop over the parameters and print each of them out
    # so they can be manually set
    for p in sorted(params.keys()):
        print "\t %s: %f" % (p, bestParams[p])

    # show a reminder message
    print "\nIMPORTANT"
    print "Now that your parameters have been searched, manually set"
    print "them and re-run this script with --optomize 0"
    
# otherwise, use the manually specified parameters
else:
    # evaluate using SVM
    clf = svm.SVC()
    clf.fit(trainX, trainY)
    
    print "SVM ON ORIGINAL DATASET"
    pred = clf.predict(testX)
    print classification_report(testY, pred)
    print("Accuracy Score: %s\n" % accuracy_score(testY, pred))

