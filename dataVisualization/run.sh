#!/bin/sh

source header.sh

echo "TRAIN & RUN CLASSIFIER"
python $prjPath/rbm.py --xTrain "$xTrain" --yTrain "$yTrain" --xTest "$xTest" --yTest "$yTest" --optimize load --visualize 0 --predOut 1 --verbose 1

echo "TRAIN & RUN REGRESSOR"
python $prjPath/multiLblReg.py --xTrain "$xTrain" --yTrain "$yTrain" --xTest "$xTest" --yTest "$yTest" -v 0 -p 1

echo "LOAD FRAMES AND APPLY OVERLAY"
python clfOverlay.py -i "$inputFrames" -c "$yTest" -o output -p "rbmPred.csv" -r "multiPred.csv" -a 100

