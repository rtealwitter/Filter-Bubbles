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
def evolve(A, s, steps, updateA=lambda A, z, epsilon: A, epsilon=.1, opinionform=opinionFJ, fixed=[]):
    ''' evolve opinions for steps time steps:
        updateA is how to updateA at each step (default to none)
        epsilon is weight to change in friendship and input to updateA
        opinionform is how to udpate opinions
        fixed is which nodes keep their innate opinions '''
    A = A.copy()
    z = [s.copy()]
    for step in range(steps):
        z.append(opinionform(A, s, z[-1]))
        A = updateA(A, z[-1], epsilon)
        for u in fixed:
            z[-1][u] = s[u]
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
            change = min(A[u,furthest], epsilon)
            A[u,furthest] = A[furthest,u] = A[furthest,u] - change
            A[u,closest] = A[closest,u] = A[closest,u] + change
    return A

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
