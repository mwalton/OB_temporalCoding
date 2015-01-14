import matplotlib.pyplot as plt
from Image import NEAREST
from matplotlib.cm import cmap_d
#import pylab as pl

def accuracy(target, prediction):
    correct = (target == prediction)
    correct = [correct, correct]
    compare = [target, prediction]
    
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(6,8))
    
    #show confusion matrix
    ax1.imshow(compare, extent=[0,100,0,1], aspect='auto', interpolation='nearest')
    #plt.colorbar()
    ax1.set_title("Prediction vs. Target")
    #ax1.ylabel('Target label')
    #ax1.xlabel('Predicted label')
    #ax1.figure(figsize=(5,10))
    
    imgPlt = ax2.imshow(correct, extent=[0,100,0,1], aspect=50, interpolation='nearest')
    imgPlt.set_cmap('RdYlGn')
    ax2.set_title("Accuracy")
    
    
    plt.show()