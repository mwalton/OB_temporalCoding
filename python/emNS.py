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
ap.add_argument("-c", "--concentration", default="none",
    help = "add concentration series data to the plot")
ap.add_argument("-s", "--start", type=int, default=0,
    help = "the timpoint in the series to begin measurement")
ap.add_argument("-e", "--end", type=int, default=-1,
    help = "the timpoint in the series to end measurement")
args = vars(ap.parse_args())

data = np.genfromtxt(args["input"], delimiter=",", dtype="float32", skip_header=1)

if (args["end"] == -1):
    end = np.shape(data)[0] - 1
else:
    end = args["end"]
    
start = args["start"]

data = np.absolute(data)

ctx_pred = data[start:end,1]
f_pred = data[start:end,0]
target = data[start:end,2]

ctx_accuracy = (target == ctx_pred)
f_accuracy = (target == f_pred)

if (not args["concentration"] == "none"):
    c = np.genfromtxt(args["concentration"], delimiter=",", dtype="float32")
else:
    c = np.zeros((0,0))

print("PREDICTION FROM CORTEX")
print classification_report(target, ctx_pred)
print("Accuracy Score: %s\n" % accuracy_score(target, ctx_pred))

print("PREDICTION FROM FIBERS")
print classification_report(target, f_pred)
print("Accuracy Score: %s\n" % accuracy_score(target, f_pred))

if (args["visualize"] == 1):
    plot.accuracy(target, ctx_pred, label="Cortex", c=c)
    plot.accuracy(target, f_pred, label="Fibers", c=c)
    plot.show()

