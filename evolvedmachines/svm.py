# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 16:41:42 2014

@author: michaelwalton
"""
import numpy as np
import scipy as sp

from data import olfactoryDataset
from sklearn import svm
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import RandomizedSearchCV

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

"""
parameterEstimation options: none, exhaustive, random, fixed_range
rand_iter: number of samples in the parameter space to sample in random estimation mode
"""
class SVC:
    def __init__(self, addNoise = False, standardize = True):
        self.addNoise = addNoise
        self.standardize = standardize
        self.target_names = ['odorant 0', 'odorant 1', 'odorant 2', 'odorant 3']
    
    def loadSet(self, train, test):
        self.data = olfactoryDataset()
        self.data.loadTraining(train)
        self.data.loadTesting(test)
        if (self.addNoise):
            self.data.addNoise()
        if (self.standardize):
            self.data.standardize()
    
    def loadData(self, folder):
        self.data = olfactoryDataset()
        self.data.load(folder)
        if (self.addNoise):
            self.data.addNoise()
        if (self.standardize):
            self.data.standardize()
    
    def predict(self):
        self.pred = self.clf.predict(self.data.test_a)
    
    def fit(self, parameterEstimation = 'none', rand_iter=10):
        if (parameterEstimation == 'none'):
            self.clf = svm.SVC()
            self.clf.fit(self.data.train_a, self.data.train_target)
        else:
            # Build multiple classifiers based on some parameter space (random, grid, etc.)
            # assess each using stratified k-folds of test data, select the best estimator
            
            #configure stratified k-fold cross validation               
            cv = StratifiedKFold(y=self.data.train_target, n_folds=3, shuffle=True)
            
            #set the parameter grid
            if (parameterEstimation == 'exhaustive'):
                #do exhaustive gridsearch within some range
                kernel_range = ['rbf', 'linear']
                gamma_range = np.arange(start=1e-3, stop=1e-1, step=1e-3)
                C_range = np.arange(1,1000)
                param_grid = dict(kernel=kernel_range, gamma=gamma_range, C=C_range)
                grid = GridSearchCV(svm.SVC(C=1), param_grid=param_grid, cv=cv)
                                
            elif(parameterEstimation == 'random'):
                #randomly sample values within some distribution of params
                kernel_range = ['rbf', 'linear']
                class_weight_range = ['auto', None]
                gamma_range =  sp.stats.expon(scale=.1)
                C_range = sp.stats.expon(scale=100)
                param_dist = dict(kernel=kernel_range, gamma=gamma_range, C=C_range,
                                  class_weight=class_weight_range)
                
                grid = RandomizedSearchCV(svm.SVC(), param_distributions=param_dist,
                                          cv=cv, n_iter=rand_iter)
                
            elif(parameterEstimation == 'fixed_range'):
                #exhaustive search on an explictly defined dictionary of params
                param_grid = [{'kernel': ['rbf'], 'gamma': [1e-1, 1e-5],
                                'C': [1, 10, 100, 1000]},
                                {'kernel': ['linear'],
                                'C': [1, 10, 100, 1000]}]
                grid = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=cv)
        
            #fit and validate until we find the best estimator within the range
            grid.fit(self.data.train_a, self.data.train_target)
            
            #show the resulting estimators
            for params, mean_score, scores in grid.grid_scores_:
                print("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() / 2, params))
        
            self.clf = grid.best_estimator_
            
    def classification_report(self):
        return classification_report(self.data.test_target, self.pred, target_names=self.target_names)
    
    def accuracy_score(self):
        return accuracy_score(self.data.test_target, self.pred)
        