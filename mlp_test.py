from evolvedmachines.mlp import NN

dataFolder = "data/Otrain_4Otest/" #folders: Otrain_4Otest, OBGtrain_4OBGtest, Otrain_4OBGtest

network = NN()
network.loadData(dataFolder)
network.buildNetwork(hiddenLayer=50)
network.runTraining(n=20)