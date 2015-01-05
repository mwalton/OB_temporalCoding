# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 16:41:42 2014

@author: michaelwalton
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl

from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import confusion_matrix

expType = 3
preprocess = True
tuneHyperparams = False

###############################################################################
# Pick a dataset
# As the project grows, this should be replaced by a line arg.
# to set a containing folder then run on the data in that dir
if (expType == 0):
    train_conc_file = "data/Otrain_4Otest/train_c.csv"
    train_actv_file = "data/Otrain_4Otest/train_a.csv"
    test_conc_file = "data/Otrain_4Otest/test_c.csv"
    test_actv_file = "data/Otrain_4Otest/test_a.csv"
elif (expType == 1):
    train_conc_file = "data/OBGtrain_4OBGtest/train_c.csv"
    train_actv_file = "data/OBGtrain_4OBGtest/train_a.csv"
    test_conc_file = "data/OBGtrain_4OBGtest/test_c.csv"
    test_actv_file = "data/OBGtrain_4OBGtest/test_a.csv"
elif (expType == 2):
    train_conc_file = "data/Otrain_4OBGtest/train_c.csv"
    train_actv_file = "data/Otrain_4OBGtest/train_a.csv"
    test_conc_file = "data/Otrain_4OBGtest/test_c.csv"
    test_actv_file = "data/Otrain_4OBGtest/test_a.csv"
else:
    train_conc_file = "data/Otrain_4OBGtest/train_c.csv"
    train_actv_file = "data/Otrain_4OBGtest/train_a_rs.csv"
    test_conc_file = "data/Otrain_4OBGtest/test_c.csv"
    test_actv_file = "data/Otrain_4OBGtest/test_a_rs.csv"

###############################################################################
#load data
reader = csv.reader(open(train_conc_file,"rb"), delimiter=",")
x = list(reader)
train_c = np.array(x).astype('float')

reader = csv.reader(open(train_actv_file,"rb"), delimiter=",")
x = list(reader)
train_a = np.array(x).astype('float')

reader = csv.reader(open(test_conc_file,"rb"), delimiter=",")
x = list(reader)
test_c = np.array(x).astype('float')

reader = csv.reader(open(test_actv_file,"rb"), delimiter=",")
x = list(reader)
test_a = np.array(x).astype('float')

###############################################################################
# Clean up the data
#get max(concentration) foreach t (target is max odorant index)
train_target = np.zeros([train_c.shape[0]], dtype=float)

for i in range(train_c.shape[0]):
    maxC = 0
    for j in range(train_c.shape[1]):
        if (train_c[i][j] > maxC and j > 0):
            maxC = train_c[i][j]
            train_target[i] = j - 1
            
test_target = np.zeros([test_c.shape[0]], dtype=float)

for i in range(test_c.shape[0]):
    maxC = 0
    for j in range(test_c.shape[1]):
        if (test_c[i][j] > maxC and j > 0):
            maxC = test_c[i][j]
            test_target[i] = j - 1

###############################################################################
# Data Pre-processing
# Improves classification on 
if (preprocess):
    scaler = StandardScaler()
    train_a = scaler.fit_transform(train_a, train_target)

###############################################################################
# Train and test the SVM
# train svm classifier

if (tuneHyperparams):
    C_range = 10.0 ** np.arange(-1, 1)
    gamma_range = 10.0 ** np.arange(-1, 1)
    param_grid = dict(gamma=gamma_range, C=C_range)
    cv = StratifiedKFold(y=train_target, n_folds=3)
    grid = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=cv)
    grid.fit(train_a, train_target)
    print("Best Classifier: ", grid.best_estimator_)
    bestClf = grid.best_estimator_
    bestPred = bestClf.predict(test_a)
    
    score_dict = grid.grid_scores_
    scores = [z[1] for z in score_dict]
    scores = np.array(scores).reshape(len(C_range), len(gamma_range))
    pl.figure(10)
    pl.imshow(scores, interpolation='nearest', cmap=pl.cm.spectral)
    pl.xlabel('gamma')
    pl.ylabel('C')
    pl.colorbar()    
    pl.show()

clf = svm.SVC()
print("Default classifier: ", clf)
clf.fit(train_a, train_target)

# run the prediction
pred = clf.predict(test_a)

# test classification
correctPredictions = 0
bestClfCorrect = 0
for i in range(pred.shape[0]):
    if(pred[i] == test_target[i]):
        correctPredictions += 1.0
    if (tuneHyperparams and bestPred[i] == test_target[i]):
        bestClfCorrect += 1.0

defaultClfPerformance = correctPredictions / test_target.shape[0]
optoClfPerformance = bestClfCorrect / test_target.shape[0]

print("Default H-params", defaultClfPerformance)
print("Optomized H-params", optoClfPerformance)
print("Difference", optoClfPerformance - defaultClfPerformance)

###############################################################################
#PLOT DATA

#plot imported data and target
pl.figure(1)
plt.plot(train_c)
plt.title('Training (Odorant Concentration)')
plt.ylabel('Concentration')
plt.xlabel('Time')
plt.show()

pl.figure(2)
plt.plot(test_c)
plt.title('Testing (Odorant Concentration)')
plt.ylabel('Concentration')
plt.xlabel('Time')
plt.show()

pl.figure(3, figsize=(6,6))
plt.imshow(np.transpose(train_a))
#plt.colorbar()
plt.title('Training (Sensor Pattern)')
plt.ylabel('Activation')
plt.xlabel('Time')
plt.show()

#show confusion matrix
cm = confusion_matrix(test_target, pred)
pl.figure(7)
plt.matshow(cm)
plt.colorbar()
plt.ylabel('Target label')
plt.xlabel('Predicted label')
plt.show()

"""
pl.figure(3, figsize=(6,6))
plt.imshow(np.transpose(train_a)[:, 25:150])
#plt.colorbar()
plt.title('Training (Sensor Pattern)')
plt.ylabel('Activation')
plt.xlabel('Time')
plt.show()

pl.figure(2)
plt.plot(train_target)
plt.title('Training (Target Odorant)')
plt.ylabel('Odorant Index')
plt.xlabel('Time')
plt.show()

pl.figure(5)
plt.plot(test_target)
plt.title('Testing (Target Odorant)')
plt.ylabel('Odorant Index')
plt.xlabel('Time')
plt.show()

pl.figure(6)
plt.imshow(np.transpose(test_a))
#plt.colorbar()
plt.title('Testing (Sensor Pattern)')
plt.ylabel('Activation')
plt.xlabel('Time')
plt.show()
"""

if (tuneHyperparams):
    cm = confusion_matrix(test_target, bestPred)
    pl.figure(8)
    plt.matshow(cm)
    plt.colorbar()
    plt.ylabel('Target label')
    plt.xlabel('Predicted label')
    plt.show()
