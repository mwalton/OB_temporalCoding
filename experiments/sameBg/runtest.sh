#!/bin/sh

source header.sh

python $prjPath/rbm.py --xTrain "$xTrain" --yTrain "$yTrain" --xTest "$xTest" --yTest "$yTest" --optimize load --visualize 0 --verbose 1 --label meanBg
#python $prjPath/emNS.py --input "$nsPath" --visualize 0
