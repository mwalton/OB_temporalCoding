#!/bin/sh

source header.sh

python $prjPath/emNS.py --input "$nsPath" --visualize 1 --start 2000
