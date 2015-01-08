# -*- coding: utf-8 -*-
"""
Created on Mon Jan 5 2014

@author: michaelwalton
"""

from evolvedmachines.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl

def run(dataParentFolder):
    cells = np.zeros((10,10))
    
    for i in range(10):
        for j in range(10):
            clf = SVC(addNoise=False, standardize=True)
            
            motifs = str(i + 1)
            kds = str(j + 1)
            
            train = dataParentFolder + "m" + motifs + "a" + kds + "train/"
            test = dataParentFolder + "m" + motifs + "a" + kds + "test/"
            
            clf.loadSet(train, test)
            clf.fit(parameterEstimation='random', rand_iter=10)
            clf.predict()
            print(clf.classification_report())
            cells[i][j] = clf.accuracy_score()
            
    pl.figure(0)
    plt.imshow(cells)
    plt.title('SVC Accuracy Score')
    plt.ylabel('Affinities Per Sensor')
    plt.xlabel('Motifs per Analyte')
    plt.show()
    
if __name__ == "__main__":
    run("/Users/michaelwalton/workspace/motif_kd_manipulations_10x10/")