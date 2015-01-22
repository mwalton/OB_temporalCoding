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
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.externals import joblib
import plots as plot
import csv
#from blz.tests.common import verbose
#from sklearn.preprocessing import StandardScaler
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

def doLogisticRegression(trainX, trainY, testX, testY, optimize, pklFolder):
    picklePath = os.path.join(pklFolder,"logitModel.pkl")
    if (optimize=="gs"):
        # perform a grid search on the 'C' parameter of Logistic
        # Regression
        print "SEARCHING LOGISTIC REGRESSION HYPERPARAMS"
        params = {"C": [1.0, 10.0, 100.0]}
        start = time.time()
        gs = GridSearchCV(LogisticRegression(), params, n_jobs = 1, verbose = 1)
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
            
        logistic = gs.best_estimator_
        
        #make the 'model' folder if its not there
        if(not os.path.exists(pklFolder)):
            os.makedirs(pklFolder)
            
        # pickle this model so we can use it later
        joblib.dump(logistic, picklePath)
        
    elif (os.path.isfile(picklePath) and args["optimize"] == "load"):
        print("Loading model: %s" % picklePath)
        logistic = joblib.load(picklePath)
    elif (os.path.isfile(picklePath) and args["optimize"] == "refit"):
        print("Loading model: %s and refitting" % picklePath)
        logistic = joblib.load(picklePath)
        logistic.fit(trainX, trainY)
    
    else:
        print("Creating new Logit model with default parameters")
        logistic = LogisticRegression(C = 1.0)
        logistic.fit(trainX, trainY)
        
        #make the 'model' folder if its not there
        if(not os.path.exists(pklFolder)):
            os.makedirs(pklFolder)
            
        # pickle this model so we can use it later
        joblib.dump(logistic, picklePath)
        
    return logistic.predict(testX)
        
def doRBM(trainX, trainY, testX, testY, optimize, pklFolder):
    picklePath = os.path.join(pklFolder,"rbmModel.pkl")
    
    if (optimize=="gs"):
        # initialize the RBM + Logistic Regression pipeline
        rbm = BernoulliRBM(n_components = 200, n_iter = 40,
        learning_rate = 0.01)
        logistic = LogisticRegression(C = 1.0)
        classifier = Pipeline([("rbm", rbm), ("logistic", logistic)])
     
        # perform a grid search on the learning rate, number of
        # iterations, and number of components on the RBM and
        # C for Logistic Regression
        print "SEARCHING RBM HYPERPARAMS"
        params = {
            "rbm__learning_rate": [0.1, 0.01, 0.001],
            "rbm__n_iter": [20, 40, 80],
            "rbm__n_components": [50, 100, 200],
            "logistic__C": [1.0, 10.0, 100.0]}
     
        # perform a grid search over the parameter
        start = time.time()
        gs = GridSearchCV(classifier, params, n_jobs = 1, verbose = 2)
        gs.fit(trainX, trainY)
     
        # print diagnostic information to the user and grab the
        # best model
        print "\ndone in %0.3fs" % (time.time() - start)
        print "best score: %0.3f" % (gs.best_score_)
        print "RBM PARAMETERS"
        bestParams = gs.best_estimator_.get_params()
     
        # loop over the parameters and print each of them out
        # so they can be manually set
        for p in sorted(params.keys()):
            print "\t %s: %f" % (p, bestParams[p])
        
        clf = gs.best_estimator_    
        #clf.fit(trainX, trainY)
        #print("Accuracy Score on Validation Set: %s\n" % accuracy_score(testY, best.predict(testX)))
     
        #make the 'model' folder if its not there
        if(not os.path.exists(pklFolder)):
            os.makedirs(pklFolder)
            
        # pickle this model so we can use it later
        joblib.dump(clf, picklePath)
        
    elif (os.path.isfile(picklePath) and args["optimize"] == "load"):
        print("Loading model: %s" % picklePath)
        clf = joblib.load(picklePath)
    elif (os.path.isfile(picklePath) and args["optimize"] == "refit"):
        print("Loading model: %s and refitting" % picklePath)
        clf = joblib.load(picklePath)
        clf.fit(trainX, trainY)
    else:
        print("Creating new RBM with default parameters")
        rbm = BernoulliRBM(n_components = 200, n_iter = 40,
        learning_rate = 0.01)
        logistic = LogisticRegression(C = 1.0)
        clf = Pipeline([("rbm", rbm), ("logistic", logistic)])
        clf.fit(trainX, trainY)
        
        #make the 'model' folder if its not there
        if(not os.path.exists(pklFolder)):
            os.makedirs(pklFolder)
            
        # pickle this model so we can use it later
        joblib.dump(clf, picklePath)
        
    return clf.predict(testX)
        
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
ap.add_argument("-p", "--pickle", default="model",
    help = "pickle dump of model (output if optomize = 1, input if optomize = 0)")
ap.add_argument("-v", "--visualize", type=int, default=0,
    help = "whether or not to show visualizations after a run")
ap.add_argument("-e", "--ensemble", type=int, default=0,
    help = "in ensemble mode, run over a folder containing multiple datasets")
ap.add_argument("-s", "--saveResults", default=None,
    help = "set this flag to write the results to a file")
ap.add_argument("-l", "--label", default="",
    help = "use this to label the data frame in the output csv")
ap.add_argument("-V", "--verbose", type=int, default=0,
    help = "prints results to stdout")
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


logitPred = doLogisticRegression(trainX, trainY, testX, testY, args["optimize"], args["pickle"])
rbmPred = doRBM(trainX, trainY, testX, testY, args["optimize"], args["pickle"])

print "LOGISTIC REGRESSION PERFORMANCE"
print classification_report(testY, logitPred)
print("Accuracy Score: %s\n" % accuracy_score(testY, logitPred))

print "RBM PERFORMANCE"
print classification_report(testY, rbmPred)
print("Accuracy Score: %s\n" % accuracy_score(testY, rbmPred))

if (args["visualize"] == 1):
    plot.accuracy(testY, logitPred, "Logistic Regression", c=testC)
    plot.accuracy(testY, rbmPred, "RBM", c=testC)
    plot.show()

if (not args["saveResults"] == None):
    predictions = [(args["label"], 'logistic', logitPred), (args["label"], 'rbm', rbmPred)]
    
    with open(args["saveResults"], 'w') as csvfile:
        fieldnames = ['label', 'clf', 'accuracy_score',
                      'precision_score', 'recall_score',
                      'f1_score']
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for l,c,p in predictions:
            results = {'label': l,
                       'clf': c,
                       'accuracy_score': accuracy_score(testY, p),
                       'precision_score': precision_score(testY, p),
                       'recall_score': recall_score(testY, p),
                       'f1_score': f1_score(testY, p)}
            
            writer.writerow(results)
