#!/bin/sh

maDataPath="/Users/michaelwalton/Dropbox/Evolved Machines 2014/Machine Learning/datasets/kaggle/MotifsAff"
prjPath=/Users/michaelwalton/workspace/git/artificial-olfaction

xTrain=train/sensorActivation.csv
yTrain=train/concentration.csv
xTest=test/sensorActivation.csv
yTest=test/concentration.csv


python $prjPath/rbm.py --xTrain "$xTrain" --yTrain "$yTrain" --xTest "$xTest" --yTest "$yTest" --optimize new --visualize 0 --saveResults ma.csv --verbose 0 --recursive "$maDataPath" --label "results/"
