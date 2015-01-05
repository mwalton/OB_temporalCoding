'''
Created on Jan 5, 2015

@author: michaelwalton
'''

import numpy as np
import csv

class olfactoryDataset:
    def __init__(self):
        return
        
    def load(self, parentFolder):
        reader = csv.reader(open(parentFolder + "train_c.csv","rb"), delimiter=",")
        x = list(reader)
        train_c = np.array(x).astype('float')
        
        reader = csv.reader(open(parentFolder + "train_a.csv","rb"), delimiter=",")
        x = list(reader)
        train_a = np.array(x).astype('float')
        
        reader = csv.reader(open(parentFolder + "test_c.csv","rb"), delimiter=",")
        x = list(reader)
        test_c = np.array(x).astype('float')
        
        reader = csv.reader(open(parentFolder + "test_a.csv","rb"), delimiter=",")
        x = list(reader)
        test_a = np.array(x).astype('float')