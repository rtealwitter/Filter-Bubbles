from toolbox import *
import matplotlib.pyplot as plt
import numpy as np

# Measurement:
## Variance (we call this polarization)
## Bimodality coefficient
# Variants:
## Network administrator changes
### updateExtreme changes closest and furthest opinion by epsilon 
### updateScale
## Attack nodes
### Never change opinion, attempt to sway opinion of network
## Starting conditions
### Polarized opinions (number of 1s, -1s, 0s)
## Blocks
### Any number of blocks each with own opinion

#np.random.seed(1)

steps = 200
series, labels = [], []
p, q = .5, .2
n = 30

np.random.seed(1)
ns, opinions = [2, 2], [1, -1]
s = initialopinion(ns, opinions)
s = np.array([0,0,1,-1])
A = expectedSBM(ns=ns, p=p, q=q)
evolve(A, s, steps=steps)
