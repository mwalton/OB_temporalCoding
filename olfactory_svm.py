# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 16:41:42 2014

@author: michaelwalton
"""

#this is set up as a multiclass problem, could maybe modify to a multilabel
#problem and then treat the p(label) as a prediciton / reproduction of the
#odorant concentration, this would be really cool if that representation works

import csv
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import scipy as sp

from evolvedmachines.svm import SVC
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import RandomizedSearchCV

from sklearn.metrics import confusion_matrix

dataFolder = "data/Otrain_4Otest/" #folders: Otrain_4Otest, OBGtrain_4OBGtest, Otrain_4OBGtest

addNoise = False
standardize = True
parameterEstimation = 'none' #options: none, exhaustive, random, fixed_range
rand_iter = 10 #number of samples in the parameter space to sample in random estimation mode

###############################################################################
#Init, load data, fit, optomize (optional), predict
clf = SVC(addNoise=False, standardize=True)
clf.loadData(dataFolder)
clf.fit(parameterEstimation='none', rand_iter=10)
clf.predict()

###############################################################################
#PLOT DATA

#plot imported data and target
pl.figure(1)
plt.plot(clf.data.train_c)
plt.title('Training (Odorant Concentration)')
plt.yscale('log')
plt.ylim(1e-4, 1)
plt.ylabel('Concentration')
plt.xlabel('Time')
plt.show()

pl.figure(2)
plt.plot(clf.data.test_c)
plt.title('Testing (Odorant Concentration)')
plt.yscale('log')
plt.ylim(1e-4, 1)
plt.ylabel('Concentration')
plt.xlabel('Time')
plt.show()

pl.figure(3, figsize=(6,6))
plt.imshow(np.transpose(clf.data.train_a))
plt.title('Training (Sensor Pattern)')
plt.ylabel('Activation')
plt.xlabel('Time')
plt.show()

pl.figure(6)
plt.imshow(np.transpose(clf.data.test_a))
plt.title('Testing (Sensor Pattern)')
plt.ylabel('Activation')
plt.xlabel('Time')
plt.show()

#show confusion matrix
cm = confusion_matrix(clf.data.test_target, clf.pred)
pl.figure(7)
plt.matshow(cm)
plt.colorbar()
plt.title('SVC')
plt.ylabel('Target label')
plt.xlabel('Predicted label')
plt.show()

print("\n")
print(clf.classification_report())
print("Accuracy Score: %s\n" % clf.accuracy_score())
print("Classifier Settings: %s" % clf)
