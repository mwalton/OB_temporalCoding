#!/bin/sh

maxC1="0.9"
maxC2="0.9"

BG1Path="/Users/michaelwalton/Dropbox/Evolved Machines 2014/Machine Learning/datasets/kaggle/BG1"
BG2Path="/Users/michaelwalton/Dropbox/Evolved Machines 2014/Machine Learning/datasets/kaggle/BG2"
prjPath=/Users/michaelwalton/workspace/git/artificial-olfaction

samexTrain=$BG1Path/${maxC1}train/sensorActivation.csv
sameyTrain=$BG1Path/${maxC1}train/concentration.csv
samexTest=$BG1Path/${maxC2}test/sensorActivation.csv
sameyTest=$BG1Path/${maxC2}test/concentration.csv

diffxTrain=$BG2Path/${maxC1}train/sensorActivation.csv
diffyTrain=$BG2Path/${maxC1}train/concentration.csv
diffxTest=$BG1Path/${maxC2}test/sensorActivation.csv
diffyTest=$BG1Path/${maxC2}test/concentration.csv
