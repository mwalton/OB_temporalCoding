#!/bin/sh

source header.sh

python $prjPath/rbm.py --xTrain "$xTrain" --yTrain "$yTrain" --xTest "$xTest" --yTest "$yTest" --optimize load --visualize 0 --saveResults same.csv --verbose 1 --recursive "$sameDataPath" --label meanBg --pickle "sameMdl"

python $prjPath/rbm.py --xTrain "$xTrain" --yTrain "$yTrain" --xTest "$xTest" --yTest "$yTest" --optimize load --visualize 0 --saveResults diff.csv --verbose 1 --recursive "$diffDataPath" --label meanBg --pickle "diffMdl"

python plotResults.py -s same.csv -d diff.csv -t train_result.csv
