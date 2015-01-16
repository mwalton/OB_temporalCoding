#!/bin/sh
path=data/debug/Otrain_4Otest
dataPath="/Users/michaelwalton/Dropbox/Evolved Machines 2014/Machine Learning/datasets/debug/Otrain_4Otest"
resultsPath="/Users/michaelwalton/Dropbox/Evolved Machines 2014/Machine Learning/results/debug"
prjPath=/Users/michaelwalton/workspace/git/artificial-olfaction

xTrain=$dataPath/train/sensorActivation.csv
yTrain=$dataPath/train/concentration.csv
xTest=$dataPath/test/sensorActivation.csv
yTest=$dataPath/test/concentration.csv

nsPath=$resultsPath/tstepAccuracy.csv
