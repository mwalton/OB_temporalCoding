#!/bin/sh

emNSpath=tstepAccuracy.csv

path=data/Otrain_4Otest
xTrain=$path/train_a.csv
yTrain=$path/train_c.csv
xTest=$path/test_a.csv
yTest=$path/test_c.csv

python svm.py --xTrain $xTrain --yTrain $yTrain --xTest $xTest --yTest $yTest --optimize 1 --visualize 0
python rbm.py --xTrain $xTrain --yTrain $yTrain --xTest $xTest --yTest $yTest --optimize 1 --visualize 0
