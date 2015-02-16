###############################################################################
# Generate sample data
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt

def loadData(XPath, yPath):
    X = np.genfromtxt(XPath, delimiter=",", dtype="float32")
    y = np.genfromtxt(yPath, delimiter=",", dtype="float32")
    return (X, y)
"""
xtrainpath="/Users/michaelwalton/Dropbox/Evolved Machines 2014/Machine Learning/datasets/BGtest/BG1/0.01test/sensorActivation.csv"
ytrainpath="/Users/michaelwalton/Dropbox/Evolved Machines 2014/Machine Learning/datasets/BGtest/BG1/0.01test/concentration.csv"
xtestpath="/Users/michaelwalton/Dropbox/Evolved Machines 2014/Machine Learning/datasets/BGtest/BG2/0.19test/sensorActivation.csv"
ytestpath="/Users/michaelwalton/Dropbox/Evolved Machines 2014/Machine Learning/datasets/BGtest/BG2/0.19test/concentration.csv"
"""

xtrainpath="/Users/michaelwalton/Dropbox/Evolved Machines 2014/Machine Learning/datasets/kaggle/paul_medC_BG2/train/sensorActivation.csv"
ytrainpath="/Users/michaelwalton/Dropbox/Evolved Machines 2014/Machine Learning/datasets/kaggle/paul_medC_BG2/train/concentration.csv"
xtestpath="/Users/michaelwalton/Dropbox/Evolved Machines 2014/Machine Learning/datasets/kaggle/paul_highC_BG1/test/sensorActivation.csv"
ytestpath="/Users/michaelwalton/Dropbox/Evolved Machines 2014/Machine Learning/datasets/kaggle/paul_highC_BG1/test/concentration.csv"

(Xtrain, ytrain) = loadData(xtrainpath, ytrainpath)
(X,y) = loadData(xtestpath, ytestpath)

ytrain=ytrain[:,1:]
ytrain[ytrain<1e-3]=1e-3
ytrain=np.log10(ytrain)

y = y[:,1:]
y[y<1e-3]=1e-3
y=np.log10(y)

###############################################################################
# Fit regression model
from sklearn.svm import SVR
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_lin = SVR(kernel='linear', C=1e3)
svr_poly = SVR(kernel='poly', C=1e3, degree=2)

y_rbf = []
y_lin = []
y_poly = []

for i in range(np.shape(y)[1]):
    y_rbf.append(svr_rbf.fit(Xtrain, ytrain[:,i]).predict(X))
    y_lin.append(svr_lin.fit(Xtrain, ytrain[:,i]).predict(X))
    y_poly.append(svr_poly.fit(Xtrain, ytrain[:,i]).predict(X))

###############################################################################
# do RMSEs for the models
rbf_rmse=sqrt(mean_squared_error(y[:,0], y_rbf[0]))
lin_rmse=sqrt(mean_squared_error(y[:,0], y_lin[0]))
poly_rmse=sqrt(mean_squared_error(y[:,0], y_poly[0]))

###############################################################################
# look at the results
#import pylab as pl
#pl.scatter(X, y, c='k', label='data')
#pl.hold('on')
fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(211)
ax1.plot(y_rbf[0], c='g', label='RBF Fit')
ax1.plot(y_lin[0], c='r', label='Linear Fit')
ax1.plot(y_poly[0], c='b', label='Poly Fit')
ax1.plot(y[:,0], c='y', label='Target')
ax1.set_xlabel('Time')
ax1.set_ylabel('log[c]')
ax1.set_title('SVR O1')
ax1.legend()

ax2 = fig.add_subplot(212)
ax2.scatter(y[:,0], y_rbf[0], c='g', label=('RBF RMSE=%0.2f' % rbf_rmse))
ax2.scatter(y[:,0], y_lin[0], c='r', label=('Linear RMSE=%0.2f' % lin_rmse))
ax2.scatter(y[:,0], y_poly[0], c='b', label=('Polynomial RMSE=%0.2f' % poly_rmse))
ax2.plot(y[:,0],y[:,0],c='y')
ax2.set_xlim(np.min(y[:,0]), np.max(y[:,0]))
ax2.set_xlabel('Prediction')
ax2.set_ylabel('Actual')
ax2.legend()

plt.show()