#!/bin/sh

source header.sh

python $prjPath/multiLblReg.py --xTrain "$xTrain" --yTrain "$yTrain" --xTest "$xTest" --yTest "$yTest" --visualize 1
