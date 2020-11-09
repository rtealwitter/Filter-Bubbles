import warnings
import numpy as np
from scipy import sparse, linalg
import scipy.sparse.linalg

warnings.filterwarnings('ignore')

def buildA(row,col,data):
    n = max(col+row)+1
    return sparse.csc_matrix((data+data,(row+col,col+row)),shape=(n,n))

def laplacianL(A):
    return sparse.csgraph.laplacian(A)

def degreeD(L):
    D = linalg.block_diag(* L.diagonal())
    return sparse.csc_matrix(D)

def invDI(L):
    D = degreeD(L)
    DI = D+sparse.eye(D.shape[0])
    return sparse.linalg.inv(DI)

def invLI(L):
    LI = L+sparse.eye(L.shape[0])
    return sparse.linalg.inv(LI)

def modifyA(A, s, zt, delta, epsilon):
    row, col = A.nonzero()    
    newrow, newdata, newcol = [], [], []
    for u in range(len(s)):
        adjacent = col[row==u]
        opinion = abs(zt[adjacent]-s[u])
        vmax = adjacent[np.argmax(opinion)]
        vmin = adjacent[np.argmin(opinion)]
        if A[u,vmax] > epsilon + delta:
            change = epsilon
        else:
            change = max(A[u,vmax] - delta,0)
        newrow += [u,u]
        newcol += [vmax,vmin]
        newdata += [-change, +change]
    return A + buildA(newrow, newcol, newdata)

def standardizedmoment(X, exponent):
    mu, sigma = np.mean(X), np.std(X)
    return np.mean(np.power((X-mu)/sigma, exponent))

def bimodality(X):
    skewness = standardizedmoment(zt, 3)
    kurtosis = standardizedmoment(zt, 4)
    return (skewness**2 + 1)/kurtosis

def steps(iterations, row, col, s):
    z = [s.copy()]
    A = buildA(row, col, data=[1]*len(row))
    for t in range(iterations):
        L = laplacianL(A)
        zt = invDI(L).dot(A.dot(z[-1])+s)
        z += [zt]
        A = modifyA(A, s, zt, delta, epsilon)
    return zt

def equilibrium(row, col, s):
    A = buildA(row, col, data=[1]*len(row))
    L = laplacianL(A)
    return invLI(L).dot(s)

def randomSBM(n,p,q,row=[],col=[]):
    for u in range(n):
        for v in range(u+1,n):
            if u < n/2 and v < n/2 or u > n/2 and v > n/2:
                isedge = np.random.choice(2, 1, p=[1-p, p])
            else:
                isedge = np.random.choice(2, 1, p=[1-q, q])
            if isedge:
                row += [u]
                col += [v]
    s = np.random.default_rng().uniform(-1,1,n)
    return row, col, s

# Drawn example
row = [0,0,0,1,2,2,3,4,4,5,6]
col = [1,2,7,5,5,3,4,5,6,7,7]
s = np.matrix([.5,0,.75,0,-1,-.5,-.25,1]).transpose()
epsilon, delta = .1, .1

# random example
n, p, q = 10, .5, .1
row, col, s = randomSBM(n,p,q)

zt = steps(10, row, col, s)
zstar = equilibrium(row, col, s)

print("zt", zt)
print("zstar", zstar)

betazt = bimodality(zt)
betazstar = bimodality(zstar)

print("mean of zt", np.mean(zt))
print("mean of zstar", np.mean(zstar))
print("std of zt", np.std(zt))
print("std of zstar", np.std(zstar))
print("beta of zt", betazt)
print("beta of zstar", betazstar)


