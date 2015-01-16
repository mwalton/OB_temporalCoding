#!/bin/sh

source experiments/differentBg/header.sh

python svm.py --xTrain "$xTrain" --yTrain "$yTrain" --xTest "$xTest" --yTest "$yTest" --optimize 0 --visualize 0
python rbm.py --xTrain "$xTrain" --yTrain "$yTrain" --xTest "$xTest" --yTest "$yTest" --optimize 2 --visualize 0
#python emNS.py --input $emNSpath --visualize 0
