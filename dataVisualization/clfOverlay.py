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
ap.add_argument("-r", "--regression", required=True,
    help = "the regression to be used in the visualization")
ap.add_argument("-a", "--ampColor", type=int, default=1,
    help = "scale the color intensity with this value")
args = vars(ap.parse_args())

yTarget = np.genfromtxt(args["concentration"], delimiter=",", dtype="float32")
yTarget = np.argmax(yTarget[:,1:5], axis=1)

yPred = np.genfromtxt(args["prediction"], delimiter=",", dtype="float32")
yPred = yPred.real.astype(int)

yReg = np.genfromtxt(args["regression"], delimiter=",",dtype="float32")

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
    cVec = yReg[frameIdx]
    #cVec += 100
    cVec[cVec<1e-10] = 1e-10
    #cVec = np.log10(cVec)
    #cVec[cVec>0.0001] = 0.0001
    #cVec *= 10000
    cVec[cVec>0.1] = 0.1
    cVec[cVec<1e-6] = 1e-6
    
    #cVec = np.tanh(cVec)
    cVec *= 25500.0
    cVec[cVec<1] = 1
    cVec[cVec>255] = 255
    #print cVec
    
    #cVec *= 255.0 / cVec.max(axis=0)
    
    # do the target / classification prediction overlay
    cv2.circle(overlay, (25, 35), 20, colors[targetIdx], -1,cv2.CV_AA)
    cv2.circle(overlay, (25, 90), 20, colors[predIdx], -1,cv2.CV_AA)
    cv2.putText(overlay, "Target", (75, 45), cv2.FONT_ITALIC, 1, (255,255,255), 1,cv2.CV_AA)
    cv2.putText(overlay, "Prediction", (75, 100), cv2.FONT_ITALIC, 1, (255,255,255), 1,cv2.CV_AA)
    cv2.putText(overlay, "Readout Vector", (350, 100), cv2.FONT_ITALIC, 1, (255,255,255), 1,cv2.CV_AA)
    
    print cVec
    
    #Do the regression outlines
    cv2.ellipse(overlay, (400,50), (21,21), 0, 0, 360, (175,175,175), 1,cv2.CV_AA)
    cv2.ellipse(overlay, (450,50), (21,21), 0, 0, 360, (175,175,175), 1,cv2.CV_AA)
    cv2.ellipse(overlay, (500,50), (21,21), 0, 0, 360, (175,175,175), 1,cv2.CV_AA)
    cv2.ellipse(overlay, (550,50), (21,21), 0, 0, 360, (175,175,175), 1,cv2.CV_AA)
    
    # do the regression overlay
    cv2.circle(overlay, (400, 50), 20, (0,0,int(cVec[0])), -1,cv2.CV_AA)
    cv2.circle(overlay, (450, 50), 20, (0,int(cVec[1]),0), -1,cv2.CV_AA)
    cv2.circle(overlay, (500, 50), 20, (int(cVec[2]),0,0), -1,cv2.CV_AA)
    cv2.circle(overlay, (550, 50), 20, (0,int(cVec[3]),int(cVec[3])), -1,cv2.CV_AA)
    
    # blend with the original:
    opacity = 0.4
    cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)
    
    outpath = path.join(args["outputFolder"], f)
    cv2.imwrite(outpath, overlay)
    
    #sys.stdout.write('.')
    sys.stdout.flush()
    
print

