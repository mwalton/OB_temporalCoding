#!/bin/sh

vizPath="/Users/michaelwalton/Dropbox/Evolved Machines 2014/Machine Learning/datasets/viz"
dataPath="/Users/michaelwalton/Dropbox/Evolved Machines 2014/Machine Learning/datasets/BGtest"
prjPath=/Users/michaelwalton/workspace/git/artificial-olfaction

xTrain=$dataPath/BG1/0.09train/sensorActivation.csv
yTrain=$dataPath/BG1/0.09train/concentration.csv
xTest=$dataPath/BG2/0.09test/sensorActivation.csv
yTest=$dataPath/BG2/0.09test/concentration.csv

inputFrames="$vizPath/test/frames"
