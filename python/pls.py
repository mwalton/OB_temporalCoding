import numpy as np
from sklearn.cross_decomposition import PLSRegression
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

def loadData(XPath, yPath):
    X = np.genfromtxt(XPath, delimiter=",", dtype="float32")
    y = np.genfromtxt(yPath, delimiter=",", dtype="float32")
    return (X, y)

def scale(label):
    #label[label<1e-10]=1e-10
    return np.power(label, 0.25)
    #return np.log10(label)

def standardize(featureVector):
    scaler = StandardScaler()
    return scaler.fit_transform(featureVector)

r = 0

xtrainpath="/Users/michaelwalton/Dropbox/Evolved Machines 2014/Machine Learning/datasets/kaggle/paul_medC_BG2/train/sensorActivation.csv"
ytrainpath="/Users/michaelwalton/Dropbox/Evolved Machines 2014/Machine Learning/datasets/kaggle/paul_medC_BG2/train/concentration.csv"
xtestpath="/Users/michaelwalton/Dropbox/Evolved Machines 2014/Machine Learning/datasets/kaggle/paul_highC_BG1/test/sensorActivation.csv"
ytestpath="/Users/michaelwalton/Dropbox/Evolved Machines 2014/Machine Learning/datasets/kaggle/paul_highC_BG1/test/concentration.csv"

"""
xtrainpath="/Users/michaelwalton/Dropbox/Evolved Machines 2014/Machine Learning/datasets/BGtest/BG1/0.01train/sensorActivation.csv"
ytrainpath="/Users/michaelwalton/Dropbox/Evolved Machines 2014/Machine Learning/datasets/BGtest/BG1/0.01train/concentration.csv"
xtestpath="/Users/michaelwalton/Dropbox/Evolved Machines 2014/Machine Learning/datasets/BGtest/BG2/0.19test/sensorActivation.csv"
ytestpath="/Users/michaelwalton/Dropbox/Evolved Machines 2014/Machine Learning/datasets/BGtest/BG2/0.19test/concentration.csv"
"""

"""
rootPath='/Users/michaelwalton/Dropbox/Evolved Machines 2014/Machine Learning/datasets/compSig'
prefix='t1'

xtrainpath=("%s/%strain/sensorActivation.csv" % (rootPath, prefix))
ytrainpath=("%s/%strain/concentration.csv" % (rootPath, prefix))
xtestpath=("%s/%stest/sensorActivation.csv" % (rootPath, prefix))
ytestpath=("%s/%stest/concentration.csv" % (rootPath, prefix))
"""

(Xtrain, ytrain) = loadData(xtrainpath, ytrainpath)
(Xtest, ytest) = loadData(xtestpath, ytestpath)

#trim off background and scale
ytrain=ytrain[:,1:]
#ytrain=scale(ytrain)
Xtrain=standardize(Xtrain)

#trim off background and scale
ytest = ytest[:,1:]
#ytest = scale(ytest)
Xtest = standardize(Xtest)

pls = PLSRegression(n_components=20)
y_pls = pls.fit(Xtrain, ytrain).predict(Xtest)

pls_rmse=[]
pls_rmse.append(sqrt(mean_squared_error(ytest[:,0], y_pls[:,0])))
pls_rmse.append(sqrt(mean_squared_error(ytest[:,1], y_pls[:,1])))
pls_rmse.append(sqrt(mean_squared_error(ytest[:,2], y_pls[:,2])))
pls_rmse.append(sqrt(mean_squared_error(ytest[:,3], y_pls[:,3])))

fig = plt.figure(figsize=(20,10))

ax1 = fig.add_subplot(241)
ax1.plot(y_pls[:,0], c='r', label='PLS Fit')
#ax1.plot(y_lin[0], c='r', label='Linear Fit')
#ax1.plot(y_poly[0], c='b', label='Poly Fit')
ax1.plot(ytest[:,0], c='grey', label='Target')
ax1.set_xlabel('Time')
ax1.set_ylabel('[c]')
#ax1.set_yscale('log')
ax1.set_title('RED')
ax1.legend()

ax2 = fig.add_subplot(242)
ax2.plot(y_pls[:,1], c='g', label='PLS Fit')
#ax2.plot(y_lin[1], c='r', label='Linear Fit')
#ax2.plot(y_poly[1], c='b', label='Poly Fit')
ax2.plot(ytest[:,1], c='grey', label='Target')
ax2.set_xlabel('Time')
#ax2.set_ylabel('log[c]')
ax2.set_title('GREEN')
ax2.legend()

ax3 = fig.add_subplot(243)
ax3.plot(y_pls[:,2], c='b', label='PLS Fit')
#ax3.plot(y_lin[2], c='r', label='Linear Fit')
#ax3.plot(y_poly[2], c='b', label='Poly Fit')
ax3.plot(ytest[:,2], c='grey', label='Target')
ax3.set_xlabel('Time')
#ax3.set_ylabel('log[c]')
ax3.set_title('BLUE')
ax3.legend()

ax4 = fig.add_subplot(244)
ax4.plot(y_pls[:,3], c='y', label='PLS Fit')
#ax4.plot(y_lin[3], c='r', label='Linear Fit')
#ax4.plot(y_poly[3], c='b', label='Poly Fit')
ax4.plot(ytest[:,3], c='grey', label='Target')
ax4.set_xlabel('Time')
#ax4.set_ylabel('log[c]')
ax4.set_title('YELLOW')
ax4.legend()

ax5 = fig.add_subplot(245)
ax5.scatter(ytest[:,0], y_pls[:,0], c='r', label=('PLS RMSE=%0.2f' % pls_rmse[0]))
#ax5.scatter(y[:,0], y_lin[0], c='r', label=('Linear RMSE=%0.2f' % lin_rmse[0]))
#ax5.scatter(y[:,0], y_poly[0], c='b', label=('Polynomial RMSE=%0.2f' % poly_rmse[0]))
ax5.plot(ytest[:,0],ytest[:,0],c='grey')
ax5.set_xlim(np.min(ytest[:,0]), np.max(ytest[:,0]))
ax5.set_xlabel('Prediction')
ax5.set_ylabel('Actual')
ax5.legend()

ax6 = fig.add_subplot(246)
ax6.scatter(ytest[:,1], y_pls[:,1], c='g', label=('PLS RMSE=%0.2f' % pls_rmse[1]))
#ax6.scatter(y[:,1], y_lin[1], c='r', label=('Linear RMSE=%0.2f' % lin_rmse[1]))
#ax6.scatter(y[:,1], y_poly[1], c='b', label=('Polynomial RMSE=%0.2f' % poly_rmse[1]))
ax6.plot(ytest[:,1],ytest[:,1],c='grey')
ax6.set_xlim(np.min(ytest[:,1]), np.max(ytest[:,1]))
ax6.set_xlabel('Prediction')
#ax6.set_ylabel('Actual')
ax6.legend()

ax7 = fig.add_subplot(247)
ax7.scatter(ytest[:,2], y_pls[:,2], c='b', label=('PLS RMSE=%0.2f' % pls_rmse[2]))
#ax7.scatter(y[:,2], y_lin[2], c='r', label=('Linear RMSE=%0.2f' % lin_rmse[2]))
#ax7.scatter(y[:,2], y_poly[2], c='b', label=('Polynomial RMSE=%0.2f' % poly_rmse[2]))
ax7.plot(ytest[:,2],ytest[:,2],c='grey')
ax7.set_xlim(np.min(ytest[:,2]), np.max(ytest[:,2]))
ax7.set_xlabel('Prediction')
#ax7.set_ylabel('Actual')
ax7.legend()

ax8 = fig.add_subplot(248)
ax8.scatter(ytest[:,3], y_pls[:,3], c='y', label=('PLS RMSE=%0.2f' % pls_rmse[3]))
#ax8.scatter(y[:,3], y_lin[3], c='r', label=('Linear RMSE=%0.2f' % lin_rmse[3]))
#ax8.scatter(y[:,3], y_poly[3], c='b', label=('Polynomial RMSE=%0.2f' % poly_rmse[3]))
ax8.plot(ytest[:,3],ytest[:,3],c='grey')
ax8.set_xlim(np.min(ytest[:,3]), np.max(ytest[:,3]))
ax8.set_xlabel('Prediction')
#ax8.set_ylabel('Actual')
ax8.legend()

plt.show()