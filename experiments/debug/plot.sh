#!/bin/sh

source header.sh

python $prjPath/rbm.py --xTrain "$xTrain" --yTrain "$yTrain" --xTest "$xTest" --yTest "$yTest" --optimize load --visualize 1 &

python $prjPath/emNS.py --input "$nsPath" --visualize 1 --concentration "$yTest"
