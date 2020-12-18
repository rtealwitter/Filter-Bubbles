import numpy as np
from scipy import sparse, linalg
import matplotlib.pyplot as plt

# Measurement
def standardizedmoment(X, exponent):
    mu, sigma = np.mean(X), np.std(X)
    return np.mean(np.power((X-mu)/sigma, exponent))
def bimodality(X):
    skewness = standardizedmoment(X, 3)
    kurtosis = standardizedmoment(X, 4)
    return min((skewness**2 + 1)/kurtosis, 1)
def polarization(X):
    return np.var(X)

# Initialization
def initialopinion(ns, opinions):
    ''' initialopinion([1,2,3], [-1, 0, 1]) -> [-1,0,0,1,1,1] '''
    s = []
    for i in range(len(opinions)):
        s += [opinions[i]]*ns[i]
    return np.array(s)
def randomopinion(n):
    return (2*np.random.random((1,n))-1)[0]
def normalopinion(n, mean=0, var=1):
    unscaled = np.random.normal(loc=mean, scale=var, size=n)
    return unscaled/ max(abs(unscaled))
def randommatrix(n, p):
    A = np.zeros((n,n))
    for u in range(n):
        for v in range(u+1, n):
            isedge = np.random.choice(2, 1, p=[1-p, p])
            if isedge:
                A[u,v] = 1
    return A + A.transpose()
def group(ns, i):
    summand = 0
    for gnum in range(len(ns)):
        if i >= summand and i < summand + ns[gnum]:
            return gnum
        summand += ns[gnum]
    raise ValueError('No group!')
def randomSBM(ns,p,q):
    ''' build matrix with len(ns) blocks, p chance of intrablock
    edge, and q chance of interblock edge '''
    n = sum(ns)
    A = np.zeros((n,n))
    for u in range(n):
        for v in range(u+1, n):
            probability = p if group(ns, u) == group(ns,v) else q
            isedge = np.random.choice(2, 1, p=[1-probability, probability]).item(0)
            if isedge: A[u,v] = 1
    return A + A.transpose()
def expectedSBM(ns,p,q):
    n = sum(ns)
    A = np.zeros((n,n))
    for u in range(n):
        for v in range(u+1, n):
            if group(ns, u) == group(ns,v):
                A[u,v]=p
            else:
                A[u,v]=q
    return A + A.transpose()
def addattacker(A, s, opinion, p=.1):
    ''' add new node with opinion and p probability of any given edge '''
    n = len(A)
    newcol, newrow  = np.zeros((n,1)), np.zeros((1,n+1))
    A = np.vstack((newrow, np.hstack((newcol, A))))
    for u in range(n):
        if np.random.choice(2, 1, p=[1-p, p]):
            A[u,0] = A[0,u] = 1
    return A, np.insert(s, 0, opinion)

# Opinion formation
def opinionFJ(A, s, z):
    ''' average friends opinion with innate opinion by weight '''
    L = sparse.csgraph.laplacian(A) 
    D = linalg.block_diag(* L.diagonal())
    I = np.eye(len(D))
    invDI = np.linalg.inv(D + np.eye(len(D)))
    return invDI.dot(A.dot(z)+s)
def opinionDG(A, s, z):
    ''' average friends opinion with current opinion '''
    L = sparse.csgraph.laplacian(A) 
    D = linalg.block_diag(* L.diagonal())
    invDI = np.linalg.inv(D + np.eye(len(D)))
    return invDI.dot(A.dot(z)+z)

# Dynamics over time
def equilibrium(A, s, check=True):
    ''' calculate equilibrium opinion given network A and
    innate opinions s '''
    L = sparse.csgraph.laplacian(A)
    invLI = np.linalg.inv(L + np.eye(len(L)))
    z = invLI.dot(s)
    if check:
        z1 = evolve(A, s, steps=1000)[-1]
        assert np.allclose(z, z1)
    return z
def evolve(A, s, steps, updateA=lambda A, z, epsilon: A, epsilon=.1, opinionform=opinionFJ, fixed=[], verbose=False, friends=False):
    ''' evolve opinions for steps time steps:
        updateA is how to updateA at each step (default to none)
        epsilon is weight to change in friendship and input to updateA
        opinionform is how to update opinions
        fixed is which nodes keep their innate opinions '''
    originalA = A
    A = A.copy()
    z = [s.copy()]
    initial_nonzero = np.count_nonzero(A)
    friendships = [1]
    
    for step in range(steps):
        A = updateA(A, z[-1], epsilon)
        for u in fixed:
            z[-1][u] = s[u]
        z.append(opinionform(A, s, z[-1]))
        friendships.append(np.count_nonzero(np.around(A, 5))/initial_nonzero)
    #assert np.allclose(np.count_nonzero(A), np.count_nonzero(originalA))
    #assert np.allclose(np.sum(A), np.sum(originalA))
    if verbose:
        print(s)
        print(np.around(z[-1], 3))
        print(np.count_nonzero(originalA))
        print(np.sum(originalA))
        print(np.around(originalA, 3))
        print(np.count_nonzero(A))
        print(np.sum(A))
        print(np.around(A, 3))
    if friends:
        return z, friendships
    return z

# Network administrator changes
def updateExtreme(A, z, epsilon):
    ''' update friendships by adding change to closest opinion
    friend and subtracting change from furthest opinion friend;
    change is the smaller of epsilon and the furthest friendship weight '''
    n = len(A)
    for u in range(n):
        adjacent = np.nonzero(A[u,])[0]
        adjacentdiff = abs(z[adjacent] - z[u])
        if len(adjacent) > 1:
            closest = adjacent[np.argmin(adjacentdiff)]
            furthest = adjacent[np.argmax(adjacentdiff)]            
            #while (A[u,furthest] - epsilon) <= 0 and furthest != closest:
            #    adjacentdiff[np.where(adjacent==furthest)] = -np.Inf
            #    furthest = adjacent[np.argmax(adjacentdiff)]
            change = min(epsilon, A[u,furthest])
            A[u,furthest] = A[furthest,u] = A[furthest,u] - change
            A[u,closest] = A[closest,u] = A[closest,u] + change
    return A

def updateScale(A, z, epsilon):
    n = len(A)
    newA = A.copy()
    for u in range(n):
        if np.count_nonzero(A[u,]) > 1:
            update = (A[u,]>0)*(2 - abs(z - z[u]))*epsilon
            normalization = np.count_nonzero(A[u,])/np.sum(A[u,] + update)
            newA[u,] = (A[u,] + update)*normalization
    return newA

# Visualize
def visualize(series, labels, filename, title='Measures of Social Cohesion by Time Steps'):
    ''' visualize the polarization and bimodality of series '''
    steps = len(series[0])
    x = np.array(range(1, steps+1))
    for i in range(len(series)):
        pseriesi, bseriesi = [], []
        for step in range(steps):
            pseriesi.append(polarization(series[i][step]))
            bseriesi.append(bimodality(series[i][step]))
        plt.plot(x, pseriesi, label='{} (Polarization)'.format(labels[i]))
        plt.plot(x, bseriesi, label='{} (Bimodality)'.format(labels[i]))
    plt.ylim(bottom=0)
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',borderaxespad=0)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def visualize2(series, labels, filename, title='Measures of Social Cohesion by Time Steps', legend=False, ylabel=None):
    ''' visualize the polarization and bimodality of series '''
    steps = len(series[0])
    x = np.array(range(1, steps+1))
    for i in range(len(series)):
        plt.plot(x, series[i], label='{}'.format(labels[i]))
    plt.ylim(bottom=0, top=1)
    plt.title(title)
    if legend: plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',borderaxespad=0)
    if ylabel != None: plt.ylabel(ylabel)
    plt.xlabel('Iterations')
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

import csv

# Read real world data
def readReddit(path):
    # load adjacency matrix
    n_reddit=556
    A = np.zeros([n_reddit, n_reddit])
    z_dict={i:[] for i in range(n_reddit)}

    with open(path+'/edges_reddit.txt','r') as f:
        reader=csv.reader(f,delimiter='\t')
        for u,v in reader:
            A[int(u)-1,int(v)-1] += 1
            A[int(v)-1,int(u)-1] += 1

    # load opinions
    with open(path+'/reddit_opinion.txt', 'r') as f:
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
def readTwitter(path):
    # load adjacency matrix
    n_twitter=548
    A = np.zeros([n_twitter, n_twitter])
    z_dict={i:[] for i in range(n_twitter)}

    with open(path+'/edges_twitter.txt','r') as f:
        reader=csv.reader(f,delimiter='\t')
        for u,v in reader:
            A[int(u)-1,int(v)-1] = 1
            A[int(v)-1,int(u)-1] = 1

    # load opinions
    with open(path+'/twitter_opinion.txt','r') as f:
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


