import climate
import theanets
import numpy as np
from sklearn.cross_validation import train_test_split

def loadData(XPath, yPath):
    X = np.genfromtxt(XPath, delimiter=",", dtype="float32")
    y = np.genfromtxt(yPath, delimiter=",", dtype="float32")
    return (X, y)

xtrainpath="/Users/michaelwalton/Dropbox/Evolved Machines 2014/Machine Learning/datasets/kaggle/paul_medC_BG2/train/sensorActivation.csv"
ytrainpath="/Users/michaelwalton/Dropbox/Evolved Machines 2014/Machine Learning/datasets/kaggle/paul_medC_BG2/train/concentration.csv"
xtestpath="/Users/michaelwalton/Dropbox/Evolved Machines 2014/Machine Learning/datasets/kaggle/paul_highC_BG1/test/sensorActivation.csv"
ytestpath="/Users/michaelwalton/Dropbox/Evolved Machines 2014/Machine Learning/datasets/kaggle/paul_highC_BG1/test/concentration.csv"

(Xtrain, ytrain) = loadData(xtrainpath, ytrainpath)
(Xtest, ytest) = loadData(xtestpath, ytestpath)

ytest=ytest[:,1:]
ytrain=ytrain[:,1:]

# split up the training data into train and validation
Xtrain, Xvalidate, ytrain, yvalidate = train_test_split(
    Xtrain, ytrain, test_size=0.25, random_state=0)

training_data = [Xtrain, ytrain]
validation_data = [Xvalidate, yvalidate]
test_data = [Xtest, ytest]

climate.enable_default_logging()

exp = theanets.Experiment(
    theanets.Regressor,
    layers=(100, 200, 100, 4),
    hidden_l1=0.1,
)

exp.train(
    training_data,
    validation_data,
    optimize='sgd',
    learning_rate=0.01,
    momentum=0.5,
)

print exp.network.predict(test_data)