# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 16:40:55 2014

@author: michaelwalton
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl

# general simulator settings
expType = 1              #which input dataset to use
inhibFreq = 20.0         #frequency of LFP oscillation in the inhibitory pop (Hz)
fs = 1000.0              #sample rate in hz
oscInhib_weight = -0.25
addNoise = False

###############################################################################
# Pick a dataset
# As the project grows, this should be replaced by a line arg.
# to set a containing folder then run on the data in that dir
if (expType == 0):
    train_conc_file = "data/Otrain_4Otest/train_c.csv"
    train_actv_file = "data/Otrain_4Otest/train_a.csv"
    test_conc_file = "data/Otrain_4Otest/test_c.csv"
    test_actv_file = "data/Otrain_4Otest/test_a.csv"
elif (expType == 1):
    train_conc_file = "data/Otrain_4OBGtest/train_c.csv"
    train_actv_file = "data/Otrain_4OBGtest/train_a.csv"
    test_conc_file = "data/Otrain_4OBGtest/test_c.csv"
    test_actv_file = "data/Otrain_4OBGtest/test_a.csv"
else:
    train_conc_file = "data/OBGtrain_4OBGtest/train_c.csv"
    train_actv_file = "data/OBGtrain_4OBGtest/train_a.csv"
    test_conc_file = "data/OBGtrain_4OBGtest/test_c.csv"
    test_actv_file = "data/OBGtrain_4OBGtest/test_a.csv"
    
###############################################################################
#load data
reader = csv.reader(open(test_actv_file,"rb"), delimiter=",")
x = list(reader)
glomeruli = np.array(x).astype('float')

###############################################################################
#precompute a 20Hz sine wave over a 1 second period with ms resolution
#20Hz -> 50ms duty cycle
t = np.arange(0, glomeruli.shape[0], 1 / fs)
inhibLFP = np.sin(2 * np.pi * inhibFreq * t)
inhibLFP /= 2
inhibLFP += 0.5

if (addNoise):
    randomRange = np.random.random_sample(fs * glomeruli.shape[0])
    inhibLFP *= randomRange
    
glomeruliMS = np.transpose(np.repeat(glomeruli, 1000, axis=0))

###############################################################################
#Apply glomerular excitation and oscillating inhibition to mitrals
mitralArray = (glomeruliMS + (inhibLFP * oscInhib_weight)) / (1 + oscInhib_weight)

#mitral LIF model settings
T       = mitralArray.shape[1] - 1                  # total time to simulate (msec)
dt      = 0.5               # simulation time step (msec)
time    = np.arange(0, T+dt, dt) # time array
t_rest  = 0                   # initial refractory time

## LIF properties
Vm      = np.zeros(len(time))    # potential (V) trace over time 
Rm      = 1                   # resistance (kOhm)
Cm      = 10                  # capacitance (uF)
tau_m   = Rm*Cm               # time constant (msec)
tau_ref = 4                   # refractory period (msec)
Vth     = 1                   # spike threshold (V)
V_spike = 1.0                 # spike delta (V)

#compute mitral spike emissions for cell[0]
## iterate over each time step
for i, t in enumerate(time):
    if t > t_rest:
        I = mitralArray[0][t]
        Vm[i] = Vm[i-1] + (-Vm[i-1] + I*Rm) / tau_m * dt
        if Vm[i] >= Vth:
            Vm[i] += V_spike
            t_rest = t + tau_ref
            
###############################################################################
#PLOT DATA

#plot imported data and target

pl.figure(num=1, figsize=(10, 30))
plt.imshow(np.transpose(glomeruli))
#plt.colorbar()
plt.title('Glomeruli')
plt.ylabel('Activation')
plt.xlabel('Time')
plt.show()

#plot the inhibition signal
pl.figure(2)
plt.plot(inhibLFP[0:1000])
plt.title('Inhibitory Population LFP')
plt.ylabel('Activation')
plt.xlabel('Time')
plt.show()

pl.figure(2, figsize=(10, 30))
plt.imshow(mitralArray[:, 1000:1300])
#plt.colorbar()
plt.title('Mitral Cells')
plt.ylabel('Activation')
plt.xlabel('Time')
plt.show()

## plot membrane potential trace  
pl.figure(3)
plt.plot(time, Vm)
plt.title('Mitral Cell [0] (LIF)')
plt.ylabel('Membrane Potential (V)')
plt.xlabel('Time (msec)')
plt.ylim([0,2])
plt.show()