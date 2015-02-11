#!/bin/sh

source header.sh

nsSame="/Users/michaelwalton/Dropbox/Evolved Machines 2014/Machine Learning/results/sameBg"
nsDiff="/Users/michaelwalton/Dropbox/Evolved Machines 2014/Machine Learning/results/diffBg"

python plotResults.py -s results/0.09same.csv -d results/0.09diff.csv -t results/0.09train_result.csv -S "$nsSame" -D "$nsDiff"
