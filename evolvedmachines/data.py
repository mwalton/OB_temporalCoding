'''
Created on Jan 5, 2015

@author: michaelwalton
'''

import numpy as np
import csv
from sklearn.preprocessing import StandardScaler
#from pybrain.datasets.supervised import SupervisedDataSet
from pybrain.datasets.classification import ClassificationDataSet

class olfactoryDataset:
    def __init__(self):
        return
        
    def load(self, parentFolder):
        reader = csv.reader(open(parentFolder + "train_c.csv","rb"), delimiter=",")
        x = list(reader)
        self.train_c = np.array(x).astype('float')
        
        reader = csv.reader(open(parentFolder + "train_a.csv","rb"), delimiter=",")
        x = list(reader)
        self.train_a = np.array(x).astype('float')
        
        reader = csv.reader(open(parentFolder + "test_c.csv","rb"), delimiter=",")
        x = list(reader)
        self.test_c = np.array(x).astype('float')
        
        reader = csv.reader(open(parentFolder + "test_a.csv","rb"), delimiter=",")
        x = list(reader)
        self.test_a = np.array(x).astype('float')
        
        # Convert the concentration labels to classes
        self.train_target = np.argmax(self.train_c[:,1:5], axis=1)
        self.test_target = np.argmax(self.test_c[:,1:5], axis=1)
        
        # store as single datasets (needed for pybrain)
        self.train_dataset = ClassificationDataSet(self.train_a.shape[1], 1, nb_classes=4)
        self.train_dataset.setField('input', self.train_a)
        self.train_dataset.setField('target', self.train_c)
        
        self.test_dataset = ClassificationDataSet(self.test_a.shape[1], 1, nb_classes=4)
        self.test_dataset.setField('input', self.test_a)
        self.test_dataset.setField('target', self.train_c)
    
    def addNoise(self):
        random_state = np.random.RandomState(0)
    
        n_samples, n_features = self.train_a.shape
        self.train_a = np.c_[self.train_a, random_state.randn(n_samples, n_features)]
        
        n_samples, n_features = self.test_a.shape
        self.test_a = np.c_[self.test_a, random_state.randn(n_samples, n_features)]
        
    def standardize(self):
        scaler = StandardScaler()
        self.train_a = scaler.fit_transform(self.train_a)
        self.test_a = scaler.transform(self.test_a)