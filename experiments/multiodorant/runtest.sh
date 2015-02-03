#!/bin/sh

source header.sh

python $prjPath/multiSVR.py --xTrain "$xTrain" --yTrain "$yTrain" --xTest "$xTest" --yTest "$yTest"
