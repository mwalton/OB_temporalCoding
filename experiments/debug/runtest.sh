#!/bin/sh

source header.sh

python $prjPath/rbm.py --xTrain "$xTrain" --yTrain "$yTrain" --xTest "$xTest" --yTest "$yTest" --optimize refit --visualize 0
python $prjPath/emNS.py --input "$nsPath" --visualize 0
