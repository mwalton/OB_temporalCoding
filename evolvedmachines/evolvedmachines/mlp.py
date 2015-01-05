from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer

from pylab import ion, ioff, figure, draw, contourf, clf, show, hold, plot
from scipy import arange, meshgrid, where

from data import olfactoryDataset

class NN:
    def __init__(self):
        return
    
    def loadData(self, folder):
        self.data = olfactoryDataset()
        self.data.load(folder)
        
    def buildNetwork(self, hiddenLayer=5):
        self.fnn = buildNetwork( self.data.train_dataset.indim, hiddenLayer, self.data.train_dataset.outdim, outclass=SoftmaxLayer )
        self.trainer = BackpropTrainer( self.fnn, dataset=self.data.train_dataset, momentum=0.1, verbose=True, weightdecay=0.01)

    def runTraining(self, n=20):
        ticks = arange(-3.,6.,0.2)
        X, Y = meshgrid(ticks, ticks)
        # need column vectors in dataset, not arrays
        griddata = ClassificationDataSet(2,1, nb_classes=4)
        for i in xrange(X.size):
            griddata.addSample([X.ravel()[i],Y.ravel()[i]], [0])
        griddata._convertToOneOfMany()  # this is still needed to make the fnn feel comfy
        
        for i in range(n):
            self.trainer.trainEpochs(1)
            trnresult = percentError( self.trainer.testOnClassData(),
                                      self.data.train_dataset['class'] )
            tstresult = percentError( self.trainer.testOnClassData(
                   dataset=self.data.test_dataset), self.data.test_dataset['class'] )
        
            print "epoch: %4d" % self.trainer.totalepochs, \
                  "  train error: %5.2f%%" % trnresult, \
                  "  test error: %5.2f%%" % tstresult
            out = self.fnn.activateOnDataset(griddata)
            out = out.argmax(axis=1)  # the highest output activation gives the class
            out = out.reshape(X.shape)
            
            figure(1)
            ioff()  # interactive graphics off
            clf()   # clear the plot
            hold(True) # overplot on
            for c in [0,1,2,3]:
                here, _ = where(self.data.test_dataset['class']==c)
                plot(self.data.test_dataset['input'][here,0],self.data.test_dataset['input'][here,1],'o')
            if out.max()!=out.min():  # safety check against flat field
                contourf(X, Y, out)   # plot the contour
            ion()   # interactive graphics on
            draw()  # update the plot
            
        ioff()
        show()

