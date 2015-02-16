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
from os import listdir
#from sklearn.metrics.metrics import accuracy_score

#from blz.tests.common import verbose
#from sklearn.preprocessing import StandardScaler
#from pandas.rpy.common import load_data

def getIndependentVarIndicies(files, delimiter):
    #here we just want to be able to get all the file prefixes and return a sorted
    #array containing exactly one occurance of each prefix
    n = []
    for f in files:
        n.append(f.split(delimiter)[0])
        
    i = np.array(n)
    i = np.unique(i)
    return i

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
    #class labels are simply the analyte at max concentration
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
        # simply load the pre-fit model and use it
        print("Loading model: %s" % picklePath)
        logistic = joblib.load(picklePath)
    elif (os.path.isfile(picklePath) and args["optimize"] == "refit"):
        # load and fit to current dataset
        print("Loading model: %s and refitting" % picklePath)
        logistic = joblib.load(picklePath)
        logistic.fit(trainX, trainY)
    
    else:
        # create a new model using default parameters
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
        learning_rate = 0.01, random_state=0)
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
        learning_rate = 0.01, random_state=0)
        logistic = LogisticRegression(C = 1.0)
        clf = Pipeline([("rbm", rbm), ("logistic", logistic)])
        clf.fit(trainX, trainY)
        
        #make the 'model' folder if its not there
        if(not os.path.exists(pklFolder)):
            os.makedirs(pklFolder)
            
        # pickle this model so we can use it later
        joblib.dump(clf, picklePath)
        
    return clf.predict(testX)

def runTest(xTrain, yTrain, xTest, yTest, arguments, label="NA", index=None):
    (trainX, trainY) = loadData(xTrain, yTrain)
    (testX, testY) = loadData(xTest, yTest)
    
    # required scaling for rbm
    trainX = scale(trainX)
    testX = scale(testX)
    
    # save the test concentration series, testY becomes class labels
    testC = testY
    
    if (arguments["multiClass"] == 1):
        trainY = convertToClasses(trainY)
        testY = convertToClasses(testY)
    else:
        trainY = np.transpose(trainY)
        testY = np.transpose(testY)
    
    logitPred = doLogisticRegression(trainX, trainY, testX, testY, arguments["optimize"], arguments["pickle"])
    rbmPred = doRBM(trainX, trainY, testX, testY, arguments["optimize"], arguments["pickle"])
    
    # write the results of the training / optimization phase
    if (arguments["optimize"] == "new" or arguments["optimize"] == "gs"):
        if (not os.path.isfile(arguments["label"] + 'train_result.csv')):
            mode = 'w'
            writeH = True
        else:
            mode = 'a'
            writeH = False
        
        with open(arguments["label"] + 'train_result.csv', mode) as csvfile:
            header = ['index', 'model_fit', 'rbm_accuracy', 'logit_accuracy']
            
            label = index if not (index==None) else np.mean(testC[:,0])
        
            writer = csv.DictWriter(csvfile, fieldnames=header)
            if (writeH): writer.writeheader()
            writer.writerow({'index' : label,
                             'model_fit' : arguments["optimize"],
                             'rbm_accuracy' : accuracy_score(testY, rbmPred),
                             'logit_accuracy' : accuracy_score(testY, logitPred)
                             })
            print [label, accuracy_score(testY, rbmPred), accuracy_score(testY, logitPred)]
        
    if (arguments["verbose"] == 1):
        print "LOGISTIC REGRESSION PERFORMANCE"
        print classification_report(testY, logitPred)
        print("Accuracy Score: %s\n" % accuracy_score(testY, logitPred))
        
        print "RBM PERFORMANCE"
        print classification_report(testY, rbmPred)
        print("Accuracy Score: %s\n" % accuracy_score(testY, rbmPred))
    
    if (arguments["visualize"] == 1):
        plot.accuracy(testY, logitPred, "Logistic Regression", c=testC)
        plot.accuracy(testY, rbmPred, "RBM", c=testC)
        plot.show()

    if (arguments["predOut"] == 1):
        np.savetxt("logitPred.csv", logitPred, delimiter=",")
        np.savetxt("rbmPred.csv", rbmPred, delimiter=",")
        
    if (index==None):
        label = np.mean(testC[:,0])
    else:
        label = index
    
    predictions = [(index, 'logistic', logitPred), (index, 'rbm', rbmPred)]
 
    rets = []
    
    for l,c,p in predictions:
        ret = {'label': l,
                   'clf': c,
                   'accuracy_score': accuracy_score(testY, p),
                   'precision_score': precision_score(testY, p),
                   'recall_score': recall_score(testY, p),
                   'f1_score': f1_score(testY, p)}
        rets.append(ret)
        
    return rets
                         
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
ap.add_argument("-r", "--recursive", default="",
    help = "use this flag to iterate over all sub dirs in the indicated directory")
ap.add_argument("-V", "--verbose", type=int, default=0,
    help = "prints results to stdout")
ap.add_argument("-l", "--label", default="NA",
    help = "choose a label for the independent variable, useful if running recursive mode")
ap.add_argument("-P", "--predOut", type=int, default=0,
    help = "output the predicition vectors to file")
args = vars(ap.parse_args())

results = []

if (not args["recursive"] == ""):
    files = listdir(args["recursive"])
    indVar = getIndependentVarIndicies(files, 't')
    
    for i in indVar:
        print("Generating model for dataset " + i)
        xTrain = os.path.join(args["recursive"], i + args["xTrain"])
        yTrain = os.path.join(args["recursive"], i + args["yTrain"])
        xTest = os.path.join(args["recursive"], i + args["xTest"])
        yTest = os.path.join(args["recursive"], i + args["yTest"])
        
        result = runTest(xTrain, yTrain, xTest, yTest, arguments=args, label=args["label"], index=i)
        results.extend(result)
else:
    results = runTest(args["xTrain"], args["yTrain"], args["xTest"], args["yTest"], arguments=args, label=args["label"])

if (not args["saveResults"] == None):
    with open(args["label"] + args["saveResults"], 'w') as csvfile:
        fieldnames = ['label', 'clf', 'accuracy_score',
                      'precision_score', 'recall_score',
                      'f1_score']
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)