#!/bin/sh

label1=0.09
label2=0.19

dataPath="/Users/michaelwalton/Dropbox/Evolved Machines 2014/Machine Learning/datasets/BGtest"
prjPath=/Users/michaelwalton/workspace/git/artificial-olfaction/python

xTrain=$dataPath/BG1/${label1}train/sensorActivation.csv
yTrain=$dataPath/BG1/${label1}train/concentration.csv
xTest=$dataPath/BG2/${label2}test/sensorActivation.csv
yTest=$dataPath/BG2/${label2}test/concentration.csv

