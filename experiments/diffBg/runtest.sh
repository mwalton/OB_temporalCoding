#!/bin/sh

source pHeader.sh

python $prjPath/rbm.py --xTrain "$xTrain" --yTrain "$yTrain" --xTest "$xTest" --yTest "$yTest" --optimize new --visualize 1 --verbose 1
