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

# Goal:
# Figure showing polarization and bimodality
# of regular, updateA on SBM with biased start
# and random graph

def runexperiment(s, A, updateA, steps, filename, title, friends=False):
    series, labels = [], []

    z, friendships = evolve(A, s, updateA=updateA, steps=steps, friends=True)
    bserie = [bimodality(zi) for zi in z]
    pserie = [polarization(zi) for zi in z]
    series += [bserie, pserie]
    labels += ['Bimodality with Update', 'Polarization with Update']
    if friends:
        series.append(friendships)
        labels.append('Remaining Friends with Update')

    z, friendships = evolve(A, s, steps=steps, friends=True)
    bserie = [bimodality(zi) for zi in z]
    pserie = [polarization(zi) for zi in z]
    series += [bserie, pserie]
    labels += ['Bimodality without Update', 'Polarization without Update']
    if friends:
        series.append(friendships)
        labels.append('Remaining Friends without Update')

    visualize2(series=series, labels=labels, filename=filename, title=title)

n = 100
p, q = 30/n, 5/n
np.random.seed(42)
sbmA = randomSBM(ns=[n//2,n//2], p=p, q=q)
extremes = initialopinion([n//2, n//2], [-1, 1])
normals = np.concatenate([normalopinion(n//2, mean=.5), normalopinion(n//2, mean=-.5)])
redditA, reddits, redditz, redditn = readReddit('code/Reddit')
twitterA, twitters, twitterz, twittern = readTwitter('code/Twitter')

steps = 300
updateA=updateExtreme
# SBM with most 1 -1
runexperiment(extremes, sbmA, updateA, steps, 'graphics/extremeextreme.pdf', 'Extreme Start with Extreme Update', friends=True)

# SBM random
runexperiment(normals, sbmA, updateA, steps, 'graphics/extremerandom.pdf', 'Bimodal Random Start with Extreme Update', friends=True)

# SBM reddit
runexperiment(reddits, redditA, updateA, steps, 'graphics/extremereddit.pdf', 'Reddit Network with Extreme Update', friends=True)

# SBM twitter
runexperiment(twitters, twitterA, updateA, steps, 'graphics/extremetwitter.pdf', 'Twitter Network with Extreme Update', friends=True)

steps = 50
updateA=updateScale
# SBM with most 1 -1
runexperiment(extremes, sbmA, updateA, steps, 'graphics/scaleextreme.pdf', 'Extreme Start with Scaled Update')

# SBM random
runexperiment(normals, sbmA, updateA, steps, 'graphics/scalerandom.pdf', 'Bimodal Random Start with Scaled Update')

# SBM reddit
runexperiment(reddits, redditA, updateA, steps, 'graphics/scalereddit.pdf', 'Reddit Network with Scale Update')

# SBM twitter
runexperiment(twitters, twitterA, updateA, steps, 'graphics/scaletwitter.pdf', 'Twitter Network with Scale Update')
