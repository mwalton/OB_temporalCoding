import matplotlib.pyplot as plt
#from Image import NEAREST
#from matplotlib.cm import cmap_d

import numpy as np
#import pylab as pl

def accuracy(target, prediction, label="Classifier", c=np.zeros((0,0))):
    correct = (target == prediction)
    correct = np.array((correct, correct))
    compare = np.array((target, prediction))
    
    showC = c != np.zeros((0,0))
    
    if (showC):
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(6,10))
    else:
        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(6,8))
    
    dim = [0,compare.shape[1],0,compare.shape[0]]
    ax1.imshow(compare, extent=dim, aspect='auto', interpolation='nearest')
    ax1.set_title(label + ": Prediction vs. Target")
    
    imgPlt = ax2.imshow(correct, extent=dim, aspect='auto', interpolation='nearest')
    imgPlt.set_cmap('RdYlGn')
    ax2.set_title(label + " Prediction Accuracy")
    
    if (showC):
        ax3.plot(c)
        ax3.set_title("Concentration")
        ax3.set_yscale('log')
        ax3.set_ylim(0.02,0.7)
    
    plt.draw()
    
def show():
    plt.show()