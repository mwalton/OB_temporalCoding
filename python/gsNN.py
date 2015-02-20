import climate
import theanets
import numpy as np
from sklearn.cross_validation import train_test_split
from os import path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import platform
from scipy.ndimage.filters import gaussian_filter
from mpl_toolkits.mplot3d import Axes3D

def loadData(XPath, yPath):
    X = np.genfromtxt(XPath, delimiter=",", dtype="float32")
    y = np.genfromtxt(yPath, delimiter=",", dtype="float32")
    return (X, y)

def scale(label):
    #label[label<1e-5]=1e-5
    return np.power(label, 0.25)
    #return np.log10(label)

def standardize(featureVector):
    scaler = StandardScaler()
    return scaler.fit_transform(featureVector)


# model prediction assessment functions
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

def avg_va(yPred, yTrue, minC):
    # declare a container to hold the vector accuracy timeseries
    vecA = []
    
    # for each timestep, if at least minEvalC is present, compute vector_accuracy
    for i in range(np.shape(yTrue)[0]):
        if (np.argmax(yTrue[i,:]) > minC):
            vecA.append(vector_accuracy(yPred[i,:], yTrue[i,:]))
            
    return np.mean(vecA, dtype=float)

if(platform.system() == 'Darwin'):
    basePath="/Users/michaelwalton/Dropbox/Evolved Machines 2014/Machine Learning/datasets/kaggle"
    outputPath="/Users/michaelwalton/Dropbox/Evolved Machines 2015/Machine Learning/Models"
else:
    basePath="/home/myke/Dropbox/Evolved Machines 2014/Machine Learning/datasets/kaggle"
    outputPath="/home/myke/Dropbox/Evolved Machines 2015/Machine Learning/Models"

xtrainpath=path.join(basePath, "paul_medC_BG2/train/sensorActivation.csv")
ytrainpath=path.join(basePath, "paul_medC_BG2/train/concentration.csv")
xtestpath=path.join(basePath, "paul_highC_BG1/test/sensorActivation.csv")
ytestpath=path.join(basePath, "paul_highC_BG1/test/concentration.csv")

(Xtrain, ytrain) = loadData(xtrainpath, ytrainpath)
(Xtest, ytest) = loadData(xtestpath, ytestpath)

ytest=ytest[:,1:]
ytrain=ytrain[:,1:]

#Xtrain=standardize(Xtrain)
#Xtest=standardize(Xtest)

ytrain=scale(ytrain)
ytest=scale(ytest)

# split up the training data into train and validation
Xtrain, Xvalidate, ytrain, yvalidate = train_test_split(
    Xtrain, ytrain, test_size=0.10, random_state=0)

training_data = [Xtrain, ytrain]
validation_data = [Xvalidate, yvalidate]
test_data = [Xtest, ytest]

climate.enable_default_logging()

hiddenLayerRange = range(10,200,10)
lr_range = np.linspace(start=0.01,stop=0.5,num=25)
momentum_range = np.linspace(start=0.01,stop=0.5,num=25)

if (path.isfile(path.join(outputPath,"va.npy"))):
    va = np.load(path.join(outputPath,"va.npy"))
else:
    va = np.zeros((len(momentum_range),len(lr_range)))

    for i, m in enumerate(momentum_range):
        for j, lr in enumerate(lr_range):
            exp = theanets.Experiment(
                theanets.Regressor,
                layers=(100, 25, 4),
                #hidden_l1=0.1,
            )
            
            exp.train(
                training_data,
                validation_data,
                optimize='sgd',
                learning_rate=lr,
                momentum=m,
            )
            
            y_pls=exp.network.predict(Xtest)
            
            va[i,j] = avg_va(y_pls, ytest, 0.001)

            np.save(path.join(outputPath,"va.npy"), va)

print va

# convolve the kernel with the data
#va = scipy.signal.convolve2d(va, kernel, mode='same')
va = gaussian_filter(va, sigma=0.9, order=0)

fig = plt.figure(figsize=(20,10))

ax1 = fig.add_subplot(211)
ax1.imshow(va, extent=[hiddenLayerRange[0],hiddenLayerRange[-1],momentum_range[0],momentum_range[-1]], aspect='auto', interpolation='nearest')
ax1.set_xlabel('Momentum')
ax1.set_ylabel('Learning Rate')

X,Y=np.meshgrid(momentum_range,lr_range)

ax2 = fig.add_subplot(212, projection='3d')
ax2.plot_surface(X, Y, va,cmap='jet',rstride=1,cstride=1,linewidth=0,antialiased=True)
ax2.set_xlabel('Momentum')
ax2.set_ylabel('Learning Rate')
ax2.set_zlabel('Vector Accuracy')
#ax2.scatter(X,Y,va,c=va)

plt.show()
