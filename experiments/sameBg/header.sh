#!/bin/sh

maxC="0.1"

dataPath="/Users/michaelwalton/Dropbox/Evolved Machines 2014/Machine Learning/datasets/sameBg"
resultsPath="/Users/michaelwalton/Dropbox/Evolved Machines 2014/Machine Learning/results/sameBg"
prjPath=/Users/michaelwalton/workspace/git/artificial-olfaction

xTrain=$dataPath/${maxC}train/sensorActivation.csv
yTrain=$dataPath/${maxC}train/concentration.csv
xTest=$dataPath/${maxC}test/sensorActivation.csv
yTest=$dataPath/${maxC}test/concentration.csv

nsPath=$resultsPath/$maxC/tstepAccuracy.csv
