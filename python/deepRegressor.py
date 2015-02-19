import climate
import theanets
import numpy as np
from sklearn.cross_validation import train_test_split
from os import path
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import platform

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
else:
    basePath="/home/myke/Dropbox/Evolved Machines 2014/Machine Learning/datasets/kaggle"

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

exp = theanets.Experiment(
    theanets.Regressor,
    layers=(100, 50, 4),
    #hidden_l1=0.1,
)

if (path.isfile("mdl.pkl")):
    print "loading model from file"
    exp.load("mdl.pkl")
else:
    print "training network"
    
    t_loss=[]
    v_loss=[]
    
    
    """
    for t in trainer:
        (train,valid) = t
        t_loss.append(train['loss'])
        v_loss.append(valid['loss'])
    
    fig = plt.figure(figsize=(10,10))
    ax1 = fig.add_subplot(111)
    ax1.plot(t_loss, c='r', label='Training')
    ax1.plot(v_loss, c='b', label='Validation')
    ax1.set_xlabel('batch')
    ax1.set_ylabel('log(loss)')
    ax1.set_yscale('log')
    ax1.set_ylim(1e-2,1e-1)
    ax1.legend()
    plt.show()
    """
    """
    exp.train(
        training_data,
        validation_data,
        optimize='sgd',
        #learning_rate=0.01,
        #momentum=0.5,
    )
    """
    exp.network.save("mdl.pkl")

print exp.network.params[0]
#print next(m)

y_pls=exp.network.predict(Xtest)

print("Normalized VA: %s\n" % avg_va(y_pls, ytest, 0.001))

pls_rmse=[]
pls_rmse.append(sqrt(mean_squared_error(ytest[:,0], y_pls[:,0])))
pls_rmse.append(sqrt(mean_squared_error(ytest[:,1], y_pls[:,1])))
pls_rmse.append(sqrt(mean_squared_error(ytest[:,2], y_pls[:,2])))
pls_rmse.append(sqrt(mean_squared_error(ytest[:,3], y_pls[:,3])))

fig = plt.figure(figsize=(20,10))

ax1 = fig.add_subplot(241)
ax1.plot(y_pls[:,0], c='r', label='NN Fit')
ax1.plot(ytest[:,0], c='grey', label='Target')
ax1.set_xlabel('Time')
ax1.set_ylabel('[c]')
#ax1.set_yscale('log')
ax1.set_title('RED')
ax1.legend()

ax2 = fig.add_subplot(242)
ax2.plot(y_pls[:,1], c='g', label='NN Fit')
ax2.plot(ytest[:,1], c='grey', label='Target')
ax2.set_xlabel('Time')
ax2.set_title('GREEN')
ax2.legend()

ax3 = fig.add_subplot(243)
ax3.plot(y_pls[:,2], c='b', label='NN Fit')
#ax3.plot(y_lin[2], c='r', label='Linear Fit')
#ax3.plot(y_poly[2], c='b', label='Poly Fit')
ax3.plot(ytest[:,2], c='grey', label='Target')
ax3.set_xlabel('Time')
#ax3.set_ylabel('log[c]')
ax3.set_title('BLUE')
ax3.legend()

ax4 = fig.add_subplot(244)
ax4.plot(y_pls[:,3], c='y', label='NN Fit')
#ax4.plot(y_lin[3], c='r', label='Linear Fit')
#ax4.plot(y_poly[3], c='b', label='Poly Fit')
ax4.plot(ytest[:,3], c='grey', label='Target')
ax4.set_xlabel('Time')
#ax4.set_ylabel('log[c]')
ax4.set_title('YELLOW')
ax4.legend()

ax5 = fig.add_subplot(245)
ax5.scatter(ytest[:,0], y_pls[:,0], c='r', label=('NN nRMSE=%0.2f' % pls_rmse[0]))
#ax5.scatter(y[:,0], y_lin[0], c='r', label=('Linear RMSE=%0.2f' % lin_rmse[0]))
#ax5.scatter(y[:,0], y_poly[0], c='b', label=('Polynomial RMSE=%0.2f' % poly_rmse[0]))
ax5.plot(ytest[:,0],ytest[:,0],c='grey')
ax5.set_xlim(np.min(ytest[:,0]), np.max(ytest[:,0]))
ax5.set_xlabel('Prediction')
ax5.set_ylabel('Actual')
ax5.legend()

ax6 = fig.add_subplot(246)
ax6.scatter(ytest[:,1], y_pls[:,1], c='g', label=('NN nRMSE=%0.2f' % pls_rmse[1]))
#ax6.scatter(y[:,1], y_lin[1], c='r', label=('Linear RMSE=%0.2f' % lin_rmse[1]))
#ax6.scatter(y[:,1], y_poly[1], c='b', label=('Polynomial RMSE=%0.2f' % poly_rmse[1]))
ax6.plot(ytest[:,1],ytest[:,1],c='grey')
ax6.set_xlim(np.min(ytest[:,1]), np.max(ytest[:,1]))
ax6.set_xlabel('Prediction')
#ax6.set_ylabel('Actual')
ax6.legend()

ax7 = fig.add_subplot(247)
ax7.scatter(ytest[:,2], y_pls[:,2], c='b', label=('NN nRMSE=%0.2f' % pls_rmse[2]))
#ax7.scatter(y[:,2], y_lin[2], c='r', label=('Linear RMSE=%0.2f' % lin_rmse[2]))
#ax7.scatter(y[:,2], y_poly[2], c='b', label=('Polynomial RMSE=%0.2f' % poly_rmse[2]))
ax7.plot(ytest[:,2],ytest[:,2],c='grey')
ax7.set_xlim(np.min(ytest[:,2]), np.max(ytest[:,2]))
ax7.set_xlabel('Prediction')
#ax7.set_ylabel('Actual')
ax7.legend()

ax8 = fig.add_subplot(248)
ax8.scatter(ytest[:,3], y_pls[:,3], c='y', label=('NN nRMSE=%0.2f' % pls_rmse[3]))
#ax8.scatter(y[:,3], y_lin[3], c='r', label=('Linear RMSE=%0.2f' % lin_rmse[3]))
#ax8.scatter(y[:,3], y_poly[3], c='b', label=('Polynomial RMSE=%0.2f' % poly_rmse[3]))
ax8.plot(ytest[:,3],ytest[:,3],c='grey')
ax8.set_xlim(np.min(ytest[:,3]), np.max(ytest[:,3]))
ax8.set_xlabel('Prediction')
#ax8.set_ylabel('Actual')
ax8.legend()

plt.show()
