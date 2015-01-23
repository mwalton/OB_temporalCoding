#!/bin/sh

source header.sh

xts="$sameDataPath/$trainTarget$xTrain"
yts="$sameDataPath/$trainTarget$yTrain"
xTs="$sameDataPath/$trainTarget$xTest"
yTs="$sameDataPath/$trainTarget$yTest"

xtd="$diffDataPath/$trainTarget$xTrain"
ytd="$diffDataPath/$trainTarget$yTrain"
xTd="$diffDataPath/$trainTarget$xTest"
yTd="$diffDataPath/$trainTarget$yTest"

python $prjPath/rbm.py --xTrain "$xts" --yTrain "$yts" --xTest "$xTs" --yTest "$yTs" --optimize new --visualize 0 --pickle "sameMdl"
python $prjPath/rbm.py --xTrain "$xtd" --yTrain "$ytd" --xTest "$xTd" --yTest "$yTd" --optimize new --visualize 0 --pickle "diffMdl"
