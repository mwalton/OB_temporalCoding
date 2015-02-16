#!/bin/sh

trainTarget=0.09

sameDataPath="/Users/michaelwalton/Dropbox/Evolved Machines 2014/Machine Learning/datasets/sameBg"
diffDataPath="/Users/michaelwalton/Dropbox/Evolved Machines 2014/Machine Learning/datasets/diffBg"
sameResultsPath="/Users/michaelwalton/Dropbox/Evolved Machines 2014/Machine Learning/results/sameBg"
diffResultsPath="/Users/michaelwalton/Dropbox/Evolved Machines 2014/Machine Learning/results/diffBg"
prjPath=/Users/michaelwalton/workspace/git/artificial-olfaction/python

xTrain=train/sensorActivation.csv
yTrain=train/concentration.csv
xTest=test/sensorActivation.csv
yTest=test/concentration.csv
