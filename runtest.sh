#!/bin/sh

path=data/Otrain_4Otest
xTrain=train_a.csv
yTrain=train_c.csv
xTest=test_a.csv
yTest=test_c.csv

python rbm.py --xTrain $path/$xTrain --yTrain $path/$yTrain --xTest $path/$xTest --yTest $path/$yTest --optimize 0
python svm.py --xTrain $path/$xTrain --yTrain $path/$yTrain --xTest $path/$xTest --yTest $path/$yTest --optimize 0
