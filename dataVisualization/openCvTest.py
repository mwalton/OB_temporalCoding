import cv2
import argparse
import numpy as np
from os import listdir
from os import path
import sys

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--inputFolder", required = False,
    help = "folder containing input images")
ap.add_argument("-o", "--outputFolder", required = True,
    help = "target folder for writing modified images")
ap.add_argument("-c", "--concentration", required = True,
    help = "concentration label file")
ap.add_argument("-p", "--prediction", required=True,
    help = "the prediction to be used in this assessment")
args = vars(ap.parse_args())

yTarget = np.genfromtxt(args["concentration"], delimiter=",", dtype="float32")
yTarget = np.argmax(yTarget[:,1:5], axis=1)

yPred = np.genfromtxt(args["prediction"], delimiter=",", dtype="float32")
yPred = yPred.real.astype(int)

colors = [(0,0,255), (0,255,0), (255,0,0), (0,255,255)]

frames = listdir(args["inputFolder"])

for f in frames:
    img = cv2.imread(path.join(args["inputFolder"], f))

    # create a copy of the original:
    overlay = img.copy()
    # draw shapes:
    frameIdx = int(f.split('.')[0])
    if (frameIdx >= np.shape(yTarget)[0]):
        break
    
    
    targetIdx = yTarget[frameIdx]
    predIdx = yPred[frameIdx]
    #print t
    cv2.circle(overlay, (200, 25), 20, colors[targetIdx], -1)
    cv2.circle(overlay, (200, 75), 20, colors[predIdx], -1)
    cv2.putText(overlay, "Target", (250, 25), cv2.FONT_HERSHEY_PLAIN, 1.5, (255,255,255), 2)
    cv2.putText(overlay, "Prediction", (250, 80), cv2.FONT_HERSHEY_PLAIN, 1.5, (255,255,255), 2)
    # blend with the original:
    opacity = 0.4
    cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)
    
    outpath = path.join(args["outputFolder"], f)
    cv2.imwrite(outpath, overlay)
    
    sys.stdout.write('.')
    sys.stdout.flush()
    
print

