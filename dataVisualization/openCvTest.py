import cv2
import argparse
import numpy as np
from os import listdir
from os import path

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--inputFolder", required = False,
    help = "folder containing input images")
ap.add_argument("-o", "--outputFolder", required = True,
    help = "target folder for writing modified images")
ap.add_argument("-c", "--concentration", required = True,
    help = "concentration label file")
args = vars(ap.parse_args())

yTarget = np.genfromtxt(args["concentration"], delimiter=",", dtype="float32")
yTarget = np.argmax(yTarget[:,1:5], axis=1)

colors = [(0,0,255), (0,255,0), (255,0,0), (0,255,255)]

frames = listdir(args["inputFolder"])

for f in frames:
    img = cv2.imread(path.join(args["inputFolder"], f))

    # create a copy of the original:
    overlay = img.copy()
    # draw shapes:
    t = yTarget[int(f.split('.')[0])]
    print t
    cv2.circle(overlay, (25, 25), 12, colors[t], -1)
    # blend with the original:
    opacity = 0.4
    cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)
    
    outpath = path.join(args["outputFolder"], f)
    cv2.imwrite(outpath, overlay)

