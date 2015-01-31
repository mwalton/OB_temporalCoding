#!/bin/sh

dataPath="/Users/michaelwalton/Dropbox/Evolved Machines 2014/Machine Learning/datasets/viz"
resultsPath="/Users/michaelwalton/Dropbox/Evolved Machines 2014/Machine Learning/results/viz"
prjPath=/Users/michaelwalton/workspace/git/artificial-olfaction

xTrain=$dataPath/train/sensorActivation.csv
yTrain=$dataPath/train/concentration.csv
xTest=$dataPath/test/sensorActivation.csv
yTest=$dataPath/test/concentration.csv

#nsPath=$resultsPath/$maxC/tstepAccuracy.csv

inputFrames="$dataPath/test/frames"
