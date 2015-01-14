import numpy as np
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV
import time
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
from sklearn.cross_validation import StratifiedKFold

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
ap.add_argument("-o", "--optimize", type = int, default = 0,
    help = "whether or not a grid search should be performed")
ap.add_argument("-m", "--multiClass", type = int, default=1,
    help = "exclusive multi class or regression")
ap.add_argument("-p", "--pickle", default="models/svmModel.pkl",
    help = "pickle dump of model (output if optomize = 1, input if optomize = 0)")
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
if args["optimize"] == 1:
    #configure stratified k-fold cross validation               
    cv = StratifiedKFold(y=trainY, n_folds=4, shuffle=True)
    # perform a grid search on the 'C' and 'gamma' parameter
    # of SVM
    print "SEARCHING SVM"
    
    C_range = 2. ** np.arange(-15, 15, step=1)
    gamma_range = 2. ** np.arange(-15, 15, step=1)

    param_grid = dict(gamma=gamma_range, C=C_range)
    
    start = time.time()
    gs = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=cv, n_jobs = -1, verbose = 2)
    gs.fit(trainX, trainY)
 
    # print diagnostic information to the user and grab the
    # best model
    print "done in %0.3fs" % (time.time() - start)
    print "best score: %0.3f" % (gs.best_score_)
    print "SVM PARAMETERS"
    bestParams = gs.best_estimator_.get_params()
 
    # loop over the parameters and print each of them out
    # so they can be manually set
    print("Best Estimator: %s" % gs.best_estimator_)
    #for p in sorted(params.keys()):
    #    print "\t %s: %f" % (p, bestParams[p])
    
    print("Accuracy Score On Validation Set: %s\n" % accuracy_score(testY, gs.predict(testX)))

    # show a reminder message
    print "\nIMPORTANT"
    print "Now that your parameters have been searched, manually set"
    print "them and re-run this script with --optomize 0"
    
    joblib.dump(gs.best_estimator_, args["pickle"])
    
# otherwise, use the manually specified parameters
else:
    # evaluate using SVM
    clf = joblib.load(args["pickle"])
    clf.fit(trainX, trainY)
    
    print "SVM ON ORIGINAL DATASET"
    pred = clf.predict(testX)
    print classification_report(testY, pred)
    print("Accuracy Score: %s\n" % accuracy_score(testY, pred))

