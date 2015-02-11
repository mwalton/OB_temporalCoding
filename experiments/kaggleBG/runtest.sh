#!/bin/sh

source header.sh

echo "SAME RESULT"

python $prjPath/rbm.py --xTrain "$samexTrain" --yTrain "$sameyTrain" --xTest "$samexTest" --yTest "$sameyTest" --optimize new --visualize 0 --verbose 1

echo "DIFFERENT RESULT"

python $prjPath/rbm.py --xTrain "$diffxTrain" --yTrain "$diffyTrain" --xTest "$diffxTest" --yTest "$diffyTest" --optimize new --visualize 0 --verbose 1
