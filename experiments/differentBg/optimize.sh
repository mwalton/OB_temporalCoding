#!/bin/sh

source experiments/differentBg/header.sh

python svm.py --xTrain "$xTrain" --yTrain "$yTrain" --xTest "$xTest" --yTest "$yTest" --optimize 1 --visualize 0
python rbm.py --xTrain "$xTrain" --yTrain "$yTrain" --xTest "$xTest" --yTest "$yTest" --optimize 1 --visualize 0
