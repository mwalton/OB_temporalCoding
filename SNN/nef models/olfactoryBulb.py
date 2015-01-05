import nef
from ca.nengo.math.impl import ConstantFunction, FourierFunction, PostfixFunction
import math


# Network olfactoryBulb Start
net_olfactoryBulb = nef.Network('olfactoryBulb')

# olfactoryBulb - Nodes

Function_inputSignal_0 = ConstantFunction(1, 0.000)
net_olfactoryBulb.make_input('inputSignal', values=[Function_inputSignal_0])
net_olfactoryBulb.make('ORNs', 100, 2, tau_rc=0.020, tau_ref=0.002, max_rate=(100.0, 200.0), intercept=(-1.0, 1.0), radius=1.00)

# olfactoryBulb - Templates
nef.templates.oscillator.make(net_olfactoryBulb, name='mitralOscillator', neurons=100, dimensions=2, frequency=125.664, tau_feedback=0.100000, tau_input=0.00000, scale=1.00000, controlled=True)

# olfactoryBulb - Projections
transform = [[0.0, 0.0],
             [0.0, 0.0],
             [0.0, 0.0]]
net_olfactoryBulb.connect('ORNs', 'mitralOscillator', transform=transform)

transform = [[0.0],
             [0.0],
             [12.566371]]
net_olfactoryBulb.connect('inputSignal', 'mitralOscillator', transform=transform)


# Network olfactoryBulb End

net_olfactoryBulb.add_to_nengo()
