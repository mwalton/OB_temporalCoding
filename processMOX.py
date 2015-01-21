import numpy as np
import argparse
import plots as plot
from os import listdir
from casuarius import required

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required = True,
    help = "parent folder of MOX sensor data")
args = vars(ap.parse_args())

files = listdir(args["input"])

"""
Ethylene (l: 31 ppm, m: 46 ppm, h: 96 ppm)
carbon monoxide (l: 270 ppm, m: 397 ppm, h: 460 ppm)
Methane (l: 51 ppm, m: 115 ppm, h: 131 ppm)
"""

gasses = {"Et" : {"n" : 0, "L" : 31, "M" : 46, "H" : 96},
          "CO" : {"n" : 0, "L" : 270, "M" : 397, "H" : 460},
          "Me" : {"n" : 0, "L" : 51, "M" : 115, "H" : 131}
          }

for f in files:
    label = f.split("_")
    
    gas0_label = {label[1], label[2]}
    gas1_label = {label[3], label[4]}
    
    
    
