#!/bin/sh

sameDataPath="/Users/michaelwalton/Dropbox/Evolved Machines 2014/Machine Learning/datasets/sameBg"
diffDataPath="/Users/michaelwalton/Dropbox/Evolved Machines 2014/Machine Learning/datasets/diffBg"
sameResultsPath="/Users/michaelwalton/Dropbox/Evolved Machines 2014/Machine Learning/results/sameBg"
diffResultsPath="/Users/michaelwalton/Dropbox/Evolved Machines 2014/Machine Learning/results/diffBg"
prjPath=/Users/michaelwalton/workspace/git/artificial-olfaction

xTrain=train/sensorActivation.csv
yTrain=train/concentration.csv
xTest=test/sensorActivation.csv
yTest=test/concentration.csv

for c in 0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19
	do
		xts="$sameDataPath/$c$xTrain"
		yts="$sameDataPath/$c$yTrain"
		xTs="$sameDataPath/$c$xTest"
		yTs="$sameDataPath/$c$yTest"

		xtd="$diffDataPath/$c$xTrain"
		ytd="$diffDataPath/$c$yTrain"
		xTd="$diffDataPath/$c$xTest"
		yTd="$diffDataPath/$c$yTest"

		echo "BEGIN TRAINING IN $c mBGc"

		python $prjPath/rbm.py --xTrain "$xts" --yTrain "$yts" --xTest "$xTs" --yTest "$yTs" --optimize new --visualize 0 --pickle "sameMdl" --label "results/$c"
		python $prjPath/rbm.py --xTrain "$xtd" --yTrain "$ytd" --xTest "$xTd" --yTest "$yTd" --optimize new --visualize 0 --pickle "diffMdl" --label "results/$c"
	
		echo "BEGIN TESTING IN ALL CONCENTRATIONS IN DATASET"
		python $prjPath/rbm.py --xTrain "$xTrain" --yTrain "$yTrain" --xTest "$xTest" --yTest "$yTest" --optimize load --visualize 0 --saveResults same.csv --verbose 0 --recursive "$sameDataPath" --label "results/$c" --pickle "sameMdl"
		python $prjPath/rbm.py --xTrain "$xTrain" --yTrain "$yTrain" --xTest "$xTest" --yTest "$yTest" --optimize load --visualize 0 --saveResults diff.csv --verbose 0 --recursive "$diffDataPath" --label "results/$c" --pickle "diffMdl"
	
		echo "PLOTTING RESULTS"
		python plotResults.py -s "results/${c}same.csv" -d "results/${c}diff.csv" -t "results/${c}train_result.csv" -f "results/$c.png"
	done

