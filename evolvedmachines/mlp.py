from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer

#from scipy import arange, meshgrid

from data import olfactoryDataset

class NN:
    def __init__(self):
        return
    
    def loadData(self, folder):
        self.data = olfactoryDataset()
        self.data.load(folder)
        
    def buildNetwork(self, hiddenLayer=5, monentum=0.1, weightdecay=0.01):
        self.fnn = buildNetwork( self.data.train_dataset.indim, hiddenLayer, self.data.train_dataset.outdim, outclass=SoftmaxLayer )
        self.trainer = BackpropTrainer( self.fnn, dataset=self.data.train_dataset, momentum=monentum, verbose=True, weightdecay=weightdecay)

    def runTraining(self, n=20):
        for i in range(n):
            self.trainer.trainEpochs(1)
            trnresult = percentError( self.trainer.testOnClassData(),
                                      self.data.train_dataset['class'] )
            tstresult = percentError( self.trainer.testOnClassData(
                   dataset=self.data.test_dataset), self.data.test_dataset['class'] )
        
            print "epoch: %4d" % self.trainer.totalepochs, \
                  "  train error: %5.2f%%" % trnresult, \
                  "  test error: %5.2f%%" % tstresult

