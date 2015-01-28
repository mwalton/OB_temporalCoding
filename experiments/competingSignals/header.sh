#!/bin/sh

label=0

dataPath="/Users/michaelwalton/Dropbox/Evolved Machines 2014/Machine Learning/datasets/competingSignals/constMix_lowC"
prjPath=/Users/michaelwalton/workspace/git/artificial-olfaction

xTrain=$dataPath/${label}train/sensorActivation.csv
yTrain=$dataPath/${label}train/concentration.csv
xTest=$dataPath/${label}test/sensorActivation.csv
yTest=$dataPath/${label}test/concentration.csv

