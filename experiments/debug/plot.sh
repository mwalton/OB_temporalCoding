#!/bin/sh

source experiments/debug/header.sh

python svm.py --xTrain $xTrain --yTrain $yTrain --xTest $xTest --yTest $yTest --optimize 0 --visualize 1
python rbm.py --xTrain $xTrain --yTrain $yTrain --xTest $xTest --yTest $yTest --optimize 2 --visualize 1
python emNS.py --input $emNSpath --visualize 1
