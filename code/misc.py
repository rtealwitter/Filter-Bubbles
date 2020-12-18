from toolbox import *
import matplotlib.pyplot as plt
import numpy as np

# Things to compare:
## Baseline
## Three blocks
## Imbalanced opinion
## Attack node


np.random.seed(42)

steps = 51
series, labels = [], []
n = 1000
p, q = 30/n, 5/n


# Baseline
sbm2A = randomSBM(ns=[n//2,n//2], p=p, q=q)
normal2s = np.concatenate([normalopinion(n//2, mean=.5), normalopinion(n//2, mean=-.5)])
z = evolve(sbm2A, normal2s, steps=steps)[:-1]
series.append([bimodality(zi) for zi in z])
labels.append('Baseline')

# Attack Node
numattacks = .001*n
sbmattA, normalatts = addattacker(sbm2A, normal2s, opinion=1, p=.1)
for i in range(int(numattacks-1)):
    sbmattA, normalatts = addattacker(sbmattA, normalatts, opinion=1, p=.1)
z = evolve(sbmattA, normalatts, steps=steps, fixed=list(range(int(numattacks))))[:-1]
series.append([bimodality(zi) for zi in z])
labels.append('Attacker (.1%)')

# Attack Node
numattacks = .01*n
sbmattA, normalatts = addattacker(sbm2A, normal2s, opinion=1, p=.1)
for i in range(int(numattacks-1)):
    sbmattA, normalatts = addattacker(sbmattA, normalatts, opinion=1, p=.1)
z = evolve(sbmattA, normalatts, steps=steps, fixed=list(range(int(numattacks))))[:-1]
series.append([bimodality(zi) for zi in z])
labels.append('Attacker (1%)')


# Three blocks
sbm3A = randomSBM(ns=[n//3,n//3, n//3], p=p, q=q)
normal3s = np.concatenate([normalopinion(n//3, mean=.5), normalopinion(n//3, mean=-.5), normalopinion(n//3, mean=0)])
z = evolve(sbm3A, normal3s, steps=steps)[:-1]
series.append([bimodality(zi) for zi in z])
labels.append('Three Blocks')

# Imbalanced Opinion
sbmimA = randomSBM(ns=[n//3, 2*n//3], p=p, q=q)
normalims = np.concatenate([normalopinion(n//3, mean=.5), normalopinion(2*n//3, mean=-.5)])
z = evolve(sbmimA, normalims, steps=steps)[:-1]
series.append([bimodality(zi) for zi in z])
labels.append('Imbalanced Opinion')

visualize2(series=series, labels=labels, filename='graphics/misc.pdf', title='Miscellaneous Variations', legend=True, ylabel='Bimodality')
