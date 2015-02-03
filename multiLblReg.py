import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import argparse
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
#from sklearn.linear_model import SGDRegressor

def loadData(XPath, yPath):
    X = np.genfromtxt(XPath, delimiter=",", dtype="float32")
    y = np.genfromtxt(yPath, delimiter=",", dtype="float32")
    return (X, y)

def standardize(featureVector):
    scaler = StandardScaler()
    return scaler.fit_transform(featureVector)

def unit_vector(vector):
    """ Returns the unit vector of the input.  """
    return vector / np.linalg.norm(vector)

"""computes the angle between two vectors in radians"""
def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    angle = np.arccos(np.dot(v1_u, v2_u))
    if np.isnan(angle):
        if (v1_u == v2_u).all():
            return 0.0
        else:
            return np.pi
    return angle

"""computes the accuracy given a target and a prediciton vector"""
def vector_accuracy(v1, v2):
    #radAngle = angle_between(v1, v2)
    #return 1 - (np.degrees(radAngle) / 90)
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.dot(v1_u, v2_u)

# parsed CL args
ap = argparse.ArgumentParser()
ap.add_argument("-x", "--xTrain", required = True,
    help = "path to training feature set")
ap.add_argument("-y", "--yTrain", required = True,
    help = "path to training target set")
ap.add_argument("-X", "--xTest", required = True,
    help = "path to testing feature set")
ap.add_argument("-Y", "--yTest", required = True,
    help = "path to testing target set")
ap.add_argument("-m", "--minEvalConcentration", default=0.0,
    help = "the minimum concentration to evaluate performance")
args = vars(ap.parse_args())

(trainX, trainY) = loadData(args["xTrain"], args["yTrain"])
(testX, testY) = loadData(args["xTest"], args["yTest"])

# required scaling for SVM
trainX = standardize(trainX)
testX = standardize(testX)

# remove the background label
trainY = trainY[:,1:5]
testY = testY[:,1:5]

# init an array of regressors, one for each label
labelCount = np.shape(trainY)[1]
clfEnsemble = [LinearRegression() for _ in range(labelCount)]

# train and predict the entire series
pred = np.zeros(np.shape(testY))

for (i,clf) in enumerate(clfEnsemble):
    clf.fit(trainX, trainY[:,i])
    pred[:,i] = clf.predict(testX)

# declare a container to hold the vector accuracy timeseries
vecA = []

# for each timestep, if at least minEvalC is present, compute vector_accuracy
for i in range(np.shape(testY)[0]):
    if (np.argmax(testY[i,:]) > args["minEvalConcentration"]):
        vecA.append(vector_accuracy(pred[i,:], testY[i,:]))

print("Multi-odorant Accuracy: %s\n" % np.mean(vecA, dtype=float))

fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.plot(testY)
ax1.set_title('True Target Signals')
ax1.set_xlabel('t')
ax1.set_ylabel('log[C]')
#ax1.set_yscale('log')
#ax1.set_ylim(0.01,0.7)

ax2 = fig.add_subplot(212)
ax2.plot(pred)
ax2.set_title('Predicted Target Signals')
ax2.set_xlabel('t')
ax2.set_ylabel('log[C]')
#ax2.set_yscale('log')
#ax2.set_ylim(0.01,0.7)

plt.show()