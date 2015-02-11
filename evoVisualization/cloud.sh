#!/bin/bash

dat="/Users/michaelwalton/Dropbox/Evolved Machines 2014/Machine Learning/evo/paul/evo_01_31.txt"

python cloud.py -f "$dat" -l lbl.csv --x0 2 --x1 4 --x2 10 -Y 32
