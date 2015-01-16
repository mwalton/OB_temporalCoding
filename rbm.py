from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import BernoulliRBM
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
import os.path
import numpy as np
import argparse
import time
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
import plots as plot
from blz.tests.common import verbose
#from pandas.rpy.common import load_data

def loadData(XPath, yPath):
    X = np.genfromtxt(XPath, delimiter=",", dtype="float32")
    y = np.genfromtxt(yPath, delimiter=",", dtype="float32")
    
    #return tuple containing feature and target vectors
    return (X, y)

def scale(X, eps = 0.00001):
    # scale the data points s.t the columns of the feature space
    # (i.e the predictors) are within the range [0, 1]
    # required by RBM
    return (X - np.min(X, axis = 0)) / (np.max(X, axis = 0) + eps)

def convertToClasses(targetVector):
    return np.argmax(targetVector[:,1:5], axis=1)

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
ap.add_argument("-o", "--optimize", default = 'none',
    help = "optomization mode: 0 use default, 1 optomize, 2 use pkl model if possible")
ap.add_argument("-m", "--multiClass", type = int, default=1,
    help = "exclusive multi class or regression")
ap.add_argument("-p", "--pickle", default="model/rbmModel.pkl",
    help = "pickle dump of model (output if optomize = 1, input if optomize = 0)")
ap.add_argument("-v", "--visualize", type=int, default=0,
    help = "whether or not to show visualizations after a run")
args = vars(ap.parse_args())

(trainX, trainY) = loadData(args["xTrain"], args["yTrain"])
(testX, testY) = loadData(args["xTest"], args["yTest"])

# required scaling for rbm
trainX = scale(trainX)
testX = scale(testX)

testC = testY

if (args["multiClass"] == 1):
    trainY = convertToClasses(trainY)
    testY = convertToClasses(testY)

# check to see if a grid search should be done
if args["optimize"] == "gs":
    # perform a grid search on the 'C' parameter of Logistic
    # Regression
    print "SEARCHING LOGISTIC REGRESSION"
    params = {"C": [1.0, 10.0, 100.0]}
    start = time.time()
    gs = GridSearchCV(LogisticRegression(), params, n_jobs = -1, verbose = 1)
    gs.fit(trainX, trainY)
 
    # print diagnostic information to the user and grab the
    # best model
    print "done in %0.3fs" % (time.time() - start)
    print "best score: %0.3f" % (gs.best_score_)
    print "LOGISTIC REGRESSION PARAMETERS"
    bestParams = gs.best_estimator_.get_params()
 
    # loop over the parameters and print each of them out
    # so they can be manually set
    for p in sorted(params.keys()):
        print "\t %s: %f" % (p, bestParams[p])
        
    # initialize the RBM + Logistic Regression pipeline
    rbm = BernoulliRBM()
    logistic = LogisticRegression()
    classifier = Pipeline([("rbm", rbm), ("logistic", logistic)])
 
    # perform a grid search on the learning rate, number of
    # iterations, and number of components on the RBM and
    # C for Logistic Regression
    print "SEARCHING RBM + LOGISTIC REGRESSION"
    params = {
        "rbm__learning_rate": [0.1, 0.01, 0.001],
        "rbm__n_iter": [20, 40, 80],
        "rbm__n_components": [50, 100, 200],
        "logistic__C": [1.0, 10.0, 100.0]}
 
    # perform a grid search over the parameter
    start = time.time()
    gs = GridSearchCV(classifier, params, n_jobs = -1, verbose = 2)
    gs.fit(trainX, trainY)
 
    # print diagnostic information to the user and grab the
    # best model
    print "\ndone in %0.3fs" % (time.time() - start)
    print "best score: %0.3f" % (gs.best_score_)
    print "RBM + LOGISTIC REGRESSION PARAMETERS"
    bestParams = gs.best_estimator_.get_params()
 
    # loop over the parameters and print each of them out
    # so they can be manually set
    for p in sorted(params.keys()):
        print "\t %s: %f" % (p, bestParams[p])
    
    clf = gs.best_estimator_    
    clf.fit(trainX, trainY)
    print("Accuracy Score on Validation Set: %s\n" % accuracy_score(testY, clf.predict(testX)))
 
    #make the 'model' folder if its not there
    if(not os.path.exists(args["pickle"].split('/')[0])):
        os.makedirs(args["pickle"].split('/')[0])
        
    # pickle this model so we can use it later
    joblib.dump(clf, args["pickle"])
    
    # show a reminder message
    print "\nIMPORTANT"
    print "Now that your parameters have been searched, manually set"
    print "them and re-run this script with --search 0"
    
# otherwise, use the manually specified parameters
else:
    # evaluate using Logistic Regression and only the raw
    # features (these parameters were cross-validated)
    logistic = LogisticRegression(C = 1.0)
    logistic.fit(trainX, trainY)
    print "LOGISTIC REGRESSION PERFORMANCE"
    pred = logistic.predict(testX)
    print classification_report(testY, pred)
    print("Accuracy Score: %s\n" % accuracy_score(testY, pred)) 
 
    # initialize the RBM + Logistic Regression classifier with
    # the cross-validated parameters
    
    if (os.path.isfile(args["pickle"]) and args["optimize"] == "load"):
        print("Loading model: %s" % args["pickle"])
        classifier = joblib.load(args["pickle"])
    elif (os.path.isfile(args["pickle"]) and args["optimize"] == "refit"):
        print("Loading model: %s" % args["pickle"])
        classifier = joblib.load(args["pickle"])
        classifier.fit(trainX, trainY)
    else:
        print("Creating new model with default parameters")
        rbm = BernoulliRBM(n_components = 200, n_iter = 40,
        learning_rate = 0.01)
        logistic = LogisticRegression(C = 1.0)
        classifier = Pipeline([("rbm", rbm), ("logistic", logistic)])
        classifier.fit(trainX, trainY)
 
    # train the classifier and show an evaluation report
    
    #classifier.fit(trainX, trainY)
    print "RBM + LOGISTIC REGRESSION PERFORMANCE"
    pred = classifier.predict(testX)
    print classification_report(testY, pred)
    print("Accuracy Score: %s\n" % accuracy_score(testY, pred))

    if (args["visualize"] == 1):
        plot.accuracy(testY, pred, "RBM", c=testC)
        plot.show()
