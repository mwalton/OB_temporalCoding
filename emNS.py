import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import argparse
import plots as plot

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required = True,
    help = "csv containing target vector, ctx prediction and fiber prediction")
ap.add_argument("-v", "--visualize", type=int, default=0,
    help = "whether or not to show visualizations after a run")
args = vars(ap.parse_args())

data = np.genfromtxt(args["input"], delimiter=",", dtype="float32", skip_header=1)

target = data[:,0]
ctx_pred = data[:,1]
f_pred = data[:,2]

ctx_accuracy = (target == ctx_pred)
f_accuracy = (target == f_pred)

print("PREDICTION FROM CORTEX")
print classification_report(target, ctx_pred)
print("Accuracy Score: %s\n" % accuracy_score(target, ctx_pred))

print("PREDICTION FROM FIBERS")
print classification_report(target, f_pred)
print("Accuracy Score: %s\n" % accuracy_score(target, f_pred))

if (args["visualize"] == 1):
    plot.accuracy(target, ctx_pred, label="Cortex")
    plot.accuracy(target, f_pred, label="Fibers")
