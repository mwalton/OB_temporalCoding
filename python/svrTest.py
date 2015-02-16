###############################################################################
# Generate sample data
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def loadData(XPath, yPath):
    X = np.genfromtxt(XPath, delimiter=",", dtype="float32")
    y = np.genfromtxt(yPath, delimiter=",", dtype="float32")
    return (X, y)

def standardize(featureVector):
    scaler = StandardScaler()
    return scaler.fit_transform(featureVector)
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

X = standardize(X)
Xtrain = standardize(Xtrain)

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
rbf_rmse=[]
lin_rmse=[]
poly_rmse=[]

rbf_rmse.append(sqrt(mean_squared_error(y[:,0], y_rbf[0])))
lin_rmse.append(sqrt(mean_squared_error(y[:,0], y_lin[0])))
poly_rmse.append(sqrt(mean_squared_error(y[:,0], y_poly[0])))

rbf_rmse.append(sqrt(mean_squared_error(y[:,1], y_rbf[1])))
lin_rmse.append(sqrt(mean_squared_error(y[:,1], y_lin[1])))
poly_rmse.append(sqrt(mean_squared_error(y[:,1], y_poly[1])))

rbf_rmse.append(sqrt(mean_squared_error(y[:,2], y_rbf[2])))
lin_rmse.append(sqrt(mean_squared_error(y[:,2], y_lin[2])))
poly_rmse.append(sqrt(mean_squared_error(y[:,2], y_poly[2])))

rbf_rmse.append(sqrt(mean_squared_error(y[:,3], y_rbf[3])))
lin_rmse.append(sqrt(mean_squared_error(y[:,3], y_lin[3])))
poly_rmse.append(sqrt(mean_squared_error(y[:,3], y_poly[3])))

###############################################################################
# plot results
#import pylab as pl

fig = plt.figure(figsize=(20,10))

ax1 = fig.add_subplot(241)
ax1.plot(y_rbf[0], c='g', label='RBF Fit')
#ax1.plot(y_lin[0], c='r', label='Linear Fit')
#ax1.plot(y_poly[0], c='b', label='Poly Fit')
ax1.plot(y[:,0], c='y', label='Target')
ax1.set_xlabel('Time')
ax1.set_ylabel('log[c]')
ax1.set_title('SVR O1')
ax1.legend()

ax2 = fig.add_subplot(242)
ax2.plot(y_rbf[1], c='g', label='RBF Fit')
#ax2.plot(y_lin[1], c='r', label='Linear Fit')
#ax2.plot(y_poly[1], c='b', label='Poly Fit')
ax2.plot(y[:,1], c='y', label='Target')
ax2.set_xlabel('Time')
ax2.set_ylabel('log[c]')
ax2.set_title('SVR O2')
ax2.legend()

ax3 = fig.add_subplot(243)
ax3.plot(y_rbf[2], c='g', label='RBF Fit')
#ax3.plot(y_lin[2], c='r', label='Linear Fit')
#ax3.plot(y_poly[2], c='b', label='Poly Fit')
ax3.plot(y[:,2], c='y', label='Target')
ax3.set_xlabel('Time')
ax3.set_ylabel('log[c]')
ax3.set_title('SVR O3')
ax3.legend()

ax4 = fig.add_subplot(244)
ax4.plot(y_rbf[3], c='g', label='RBF Fit')
#ax4.plot(y_lin[3], c='r', label='Linear Fit')
#ax4.plot(y_poly[3], c='b', label='Poly Fit')
ax4.plot(y[:,3], c='y', label='Target')
ax4.set_xlabel('Time')
ax4.set_ylabel('log[c]')
ax4.set_title('SVR O4')
ax4.legend()

ax5 = fig.add_subplot(245)
ax5.scatter(y[:,0], y_rbf[0], c='g', label=('RBF RMSE=%0.2f' % rbf_rmse[0]))
#ax5.scatter(y[:,0], y_lin[0], c='r', label=('Linear RMSE=%0.2f' % lin_rmse[0]))
#ax5.scatter(y[:,0], y_poly[0], c='b', label=('Polynomial RMSE=%0.2f' % poly_rmse[0]))
ax5.plot(y[:,0],y[:,0],c='y')
ax5.set_xlim(np.min(y[:,0]), np.max(y[:,0]))
ax5.set_xlabel('Prediction')
ax5.set_ylabel('Actual')
ax5.legend()

ax6 = fig.add_subplot(246)
ax6.scatter(y[:,1], y_rbf[1], c='g', label=('RBF RMSE=%0.2f' % rbf_rmse[1]))
#ax6.scatter(y[:,1], y_lin[1], c='r', label=('Linear RMSE=%0.2f' % lin_rmse[1]))
#ax6.scatter(y[:,1], y_poly[1], c='b', label=('Polynomial RMSE=%0.2f' % poly_rmse[1]))
ax6.plot(y[:,1],y[:,1],c='y')
ax6.set_xlim(np.min(y[:,1]), np.max(y[:,1]))
ax6.set_xlabel('Prediction')
ax6.set_ylabel('Actual')
ax6.legend()

ax7 = fig.add_subplot(247)
ax7.scatter(y[:,2], y_rbf[2], c='g', label=('RBF RMSE=%0.2f' % rbf_rmse[2]))
#ax7.scatter(y[:,2], y_lin[2], c='r', label=('Linear RMSE=%0.2f' % lin_rmse[2]))
#ax7.scatter(y[:,2], y_poly[2], c='b', label=('Polynomial RMSE=%0.2f' % poly_rmse[2]))
ax7.plot(y[:,2],y[:,2],c='y')
ax7.set_xlim(np.min(y[:,2]), np.max(y[:,2]))
ax7.set_xlabel('Prediction')
ax7.set_ylabel('Actual')
ax7.legend()

ax8 = fig.add_subplot(248)
ax8.scatter(y[:,3], y_rbf[3], c='g', label=('RBF RMSE=%0.2f' % rbf_rmse[3]))
#ax8.scatter(y[:,3], y_lin[3], c='r', label=('Linear RMSE=%0.2f' % lin_rmse[3]))
#ax8.scatter(y[:,3], y_poly[3], c='b', label=('Polynomial RMSE=%0.2f' % poly_rmse[3]))
ax8.plot(y[:,3],y[:,3],c='y')
ax8.set_xlim(np.min(y[:,3]), np.max(y[:,3]))
ax8.set_xlabel('Prediction')
ax8.set_ylabel('Actual')
ax8.legend()

plt.show()