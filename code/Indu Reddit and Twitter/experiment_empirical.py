from toolbox import *
import matplotlib.pyplot as plt
import numpy as np

import numpy as np
import matplotlib.pyplot as plt

from scipy.linalg import solve
#from gurobipy import *

import networkx as nx

from time import time
from pprint import pprint
import pickle
import csv # reading network files



#####################################################
# REDDIT
#####################################################
def readReddit():

    # load adjacency matrix
    n_reddit=556
    A = np.zeros([n_reddit, n_reddit])
    z_dict={i:[] for i in range(n_reddit)}

    with open('Reddit/edges_reddit.txt','r') as f:
        reader=csv.reader(f,delimiter='\t')
        for u,v in reader:
            A[int(u)-1,int(v)-1] += 1
            A[int(v)-1,int(u)-1] += 1

    # load opinions
    with open('Reddit/reddit_opinion.txt', 'r') as f:
        reader=csv.reader(f,delimiter='\t')
        for u,v,w in reader:
            z_dict[int(u)-1].append(float(w))

    # remove nodes not ocnnected in graph
    not_connected = np.argwhere(np.sum(A,0)==0)
    A=np.delete(A,not_connected,axis=0)
    A=np.delete(A,not_connected,axis=1)
    n_reddit = n_reddit-len(not_connected)

    # create z (avg-ing posts)
    z = [np.mean(z_dict[i]) for i in range(n_reddit)]
    z=np.array(z)

    # create initial opinions from z
    L = np.diag(np.sum(A,0)) - A
    s = (L+np.eye(n_reddit)).dot(z)
    s=np.minimum(np.maximum(s,0),1)

    return A, s, z, n_reddit

#####################################################
# READ TWITTER
#####################################################
def readTwitter():

     # load adjacency matrix
     n_twitter=548
     A = np.zeros([n_twitter, n_twitter])
     z_dict={i:[] for i in range(n_twitter)}

     with open('Twitter/edges_twitter.txt','r') as f:
         reader=csv.reader(f,delimiter='\t')
         for u,v in reader:
             A[int(u)-1,int(v)-1] = 1
             A[int(v)-1,int(u)-1] = 1

     # load opinions
     with open('Twitter/twitter_opinion.txt','r') as f:
         reader=csv.reader(f,delimiter='\t')
         for u,v,w in reader:
             z_dict[int(u)-1].append(float(w))

     # remove nodes not ocnnected in graph
     not_connected = np.argwhere(np.sum(A,0)==0)
     A=np.delete(A,not_connected,axis=0)
     A=np.delete(A,not_connected,axis=1)
     n_twitter = n_twitter-len(not_connected)

     # create z (avg-ing posts)
     z = [np.mean(z_dict[i]) for i in range(n_twitter)]
     z=np.array(z)

     # create initial opinions from z
     L = np.diag(np.sum(A,0)) - A
     s = (L+np.eye(n_twitter)).dot(z)
     s=np.minimum(np.maximum(s,0),1)

     return A, s, z, n_twitter

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


steps = 700
series, labels = [], []
p, q = 1, 1
#n = n_reddit
epsilon=.1

#ns, opinions = [2,1,1,1,1], [-1,-.5,0,.5,1]
#s = initialopinion(ns, opinions)
#A = randomSBM(ns=ns, p=p, q=q)
#A,s,z,n = readReddit()
A,s,z,n = readTwitter()
series.append(evolve(A, s, steps=steps, updateA=updateExtreme, epsilon=epsilon))#, verbose=True)
labels.append('Friedkin-Johnsen Opinion')

"""
steps = 200
series, labels = [], []
p, q = 1, 1
n = 30
updateA=updateScale
epsilon=.1


#np.random.seed(1)
#ns, opinions = [3, 3], [1, -1]
#s = initialopinion(ns, opinions)
#A = randomSBM(ns=ns, p=p, q=q)
series.append(evolve(A, s, steps=steps))
labels.append('Regular')
"""

#ns, opinions = [3, 3], [1, -1]
#s = initialopinion(ns, opinions)
#A = randomSBM(ns=ns, p=p, q=q)
#A,s,z,n = readReddit()
A,s,z,n = readTwitter()
series.append(evolve(A, s, steps=steps, updateA=updateExtreme, epsilon=epsilon, opinionform=opinionDG))
labels.append('de Groot\'s Opinion')

"""#ns, opinions = [2, 4], [1, -1]
#s = initialopinion(ns, opinions)
#A = randomSBM(ns=ns, p=p, q=q)
A,s,z,n = readReddit()
series.append(evolve(A, s, steps=steps, updateA=updateA, epsilon=epsilon))
labels.append('Imbalanced Opinion (more 1s than -1s)')
"""

#np.random.seed(1)
#ns, opinions = [2, 3], [1, -1]
#s = initialopinion(ns, opinions)
#A = randomSBM(ns=ns, p=p, q=q)
#A,s,z,n = readReddit()
A,s,z,n = readTwitter()
A, s = addattacker(A, s, opinion=1, p=.1)
series.append(evolve(A, s, steps=steps, updateA=updateExtreme, epsilon=epsilon, fixed=[0]))
labels.append('Attacker Node (opinion fixed)')


visualize(series=series, labels=labels, filename='graphics/cohesion.pdf', title='Update Extreme- Twitter')
