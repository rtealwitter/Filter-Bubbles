from toolbox import *
import matplotlib.pyplot as plt
import numpy as np

steps = 200
series, labels = [], []

ns, opinions = [15, 15], [1, -1]
s = initialopinion(ns, opinions)
A = randomSBM(ns=ns, p=.8, q=.5)
series.append(evolve(A, s, steps=steps, updateA=updateExtreme, epsilon=.1))
labels.append('Update Connections')

ns, opinions = [10, 20], [1, -1]
s = initialopinion(ns, opinions)
A = randomSBM(ns=ns, p=.8, q=.5)
series.append(evolve(A, s, steps=steps))
labels.append('Imbalanced Opinion')

ns, opinions = [15, 15], [1, -1]
s = initialopinion(ns, opinions)
A = randomSBM(ns=ns, p=.8, q=.5)
A, s = addattacker(A, s, 1)
series.append(evolve(A, s, steps=steps, fixed=[0]))
labels.append('Attacker')

visualize(series=series, labels=labels, filename='graphics/cohesion.pdf')
