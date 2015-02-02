#!/bin/sh

label=0.09

dataPath="/Users/michaelwalton/Dropbox/Evolved Machines 2014/Machine Learning/datasets/BGtest"
prjPath=/Users/michaelwalton/workspace/git/artificial-olfaction

xTrain=$dataPath/BG1/${label}train/sensorActivation.csv
yTrain=$dataPath/BG1/${label}train/concentration.csv
xTest=$dataPath/BG2/${label}test/sensorActivation.csv
yTest=$dataPath/BG2/${label}test/concentration.csv

