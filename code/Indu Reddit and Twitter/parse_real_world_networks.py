import numpy as np
import matplotlib.pyplot as plt

from scipy.linalg import solve
#from gurobipy import *

import networkx as nx

from time import time
from pprint import pprint
import pickle
import csv # reading network files

###########################
# HELPERS
###########################

# q0: edge probability in group 0
# q1: edge probability in group 1
# p: edge probability in between groups

# intrinsic opinions in group i are drawn iid from N(si_mean, s_std)
# group 0 has n nodes, group 1 has m nodes
def create_A_s(q0, q1, p, s0_mean=0.25, s1_mean=0.75, s_std=0.1,n=32,m=32):
    tot = n + m

    A = np.zeros([tot,tot])
    for i in range(tot):
        for j in range(i+1, tot):
            if i > n and j > n: #q1
                A[i,j]=np.random.binomial(1,q1)
            elif j > n: # then i <=n-1 and j > n, so p
                A[i,j]=np.random.binomial(1,p)
            elif j <=n: # then i <= n-1 and j <= n-1, so q0
                A[i,j]=np.random.binomial(1,q0)
    A=A+A.T
    s = np.concatenate((np.random.normal(loc=s0_mean,scale=s_std,size=n),np.random.normal(loc=s1_mean,scale=s_std,size=m)))
    s=np.maximum(s,0)
    s=np.minimum(s,1)
    return A,s

################################################
# HELPER FUNCTIONS FOR ALTERNATING MINIMIZATION
################################################

# find z that minimizes z^T L z + |z-s|^2
# W = weight matrix for graph
# s = intrinsic opinions
def min_z(W,s):
    D = np.diag(np.sum(W,0))
    L = D - W
    n=L.shape[0]
    return solve(L+np.eye(n),s)

# find weight matrix W that minimizes z^T L z, 
# where L is graph Laplacian corresponding to W
# constrained to {W : ||W-W0|| < lam * ||W0||} where W0 is the original graph
# (lam is what proportio of edges in the original graph you are allowed to change)
# we also impose the restriction that sum(W[:,i]) = sum(W0[:,i]), 
# i.e. the degree of each vertex is conserved

# IF reduce_pls = True, then we add the term gam*||W||^2 to the objective
# as this empirically reduces polarization (encourages more connections to every vertex)

def min_w_gurobi(z, lam, W0, reduce_pls, gam):
    n = z.shape[0]
    m = Model("qcp")

    inds = [(i,j) for i in range(n) for j in range(n) if i>j]
    x=m.addVars(inds, lb=0.0, name="x")

    # obj is min \sum_{i,j} wij (zi-zj)^2
    w = {(i,j):(z[i]-z[j])**2 for i in range(n) for j in range(n) if i>j}

    obj_exp = x.prod(w)
    if reduce_pls:
        obj_exp += gam*x.prod(x)
    m.setObjective(obj_exp, GRB.MINIMIZE)
    print('added variables')
    
    # add constraints sum_j x[i,j] = di
    d = np.sum(W0,0)
    for i in range(n):
        m.addConstr(quicksum([x[(j,i)] for j in range(i+1,n)] + [x[(i,j)] for j in range(i)]) == d[i])
    print('added first constraint')
    
    # add constraint \sum_{i,j} (wij - w0ij) < lam*norm(w0)**2
    rhs = (lam*np.linalg.norm(A))**2
    
    m.addQConstr(quicksum([x[(i,j)]*x[(i,j)]-2*x[(i,j)]*W0[i,j]+W0[i,j]*W0[i,j] for i in range(n) for j in range(n) if i>j]) <= rhs)
    print('added second constraint')
    print('starting to optimize')
    m.optimize()
    
    W = np.zeros([n,n])
    for u in range(n):
        for v in range(n):
            if u>v:
                W[u,v]=x[(u,v)].X
                W[v,u] = W[u,v]
    return W

# given opinion vectors z, compute 
# polarization = variance(z)
def compute_pls(z):
    z_centered = z - np.mean(z)
    return z_centered.dot(z_centered)

# IGNORE (this doesn't really do what's intended)
def compute_obj(z,s,W,A,lam):
    # compute z^T L z + ||z-s||^2 + lam*||W-A||^2
    D = np.diag(np.sum(W,0))
    L = D - W
    return z.T.dot(L).dot(z) + np.linalg.norm(z-s)**2+lam*np.linalg.norm(W-A)**2 

################################################
# ALTERNATING MINIMIZATION
################################################

# Alternating Minimization function for network admin game
# Parameters:
# 1) A: initial graph (adjacency matrix) 

# 2) s: intrinsic opinions

# 3) lam: constraint parameter

# 4) reduce_pls: if true, implement additional L2 regularization to 
# reduce polarization (and disagreement!)
# 5) gam: regularization coefficient for reduce_pls (gam \in [0,1] seems to work)
# 6) max_iters: max number of iterations of network admin game

# RETURNS:
# pls: list of polarizations at each iteration
# z: opinions at final iteration of game
# W: adjacency matrix at final iteration of game
def am(A,s,lam,reduce_pls=False,gam=0,max_iters=2):
    # alternating minimization
    W = np.copy(A)
    z = min_z(W,s) # minimize z first

    # polarization
    pls = [compute_pls(z)]
    
    # disagreement
    disaggs = [z.T.dot(W).dot(z)]

    # LOOP: first minimize W, then minimize z
    # then decide if we should exit
    i = 0
    flag = True
    while flag:
        print('iteration: {}'.format(i))
        # minimize W
        Wnew = min_w_gurobi(z,lam,A,reduce_pls=reduce_pls,gam=gam)
        
        # minimize z
        znew = min_z(Wnew,s)

        # exit condition
        if np.maximum(np.linalg.norm(z-znew), np.linalg.norm(Wnew-W)) < 5e-1 or i > max_iters - 1:
            flag = False

        # update z,W,i,pls
        z = znew
        W = Wnew
        i=i+1
        pls.append(compute_pls(z))
        disaggs.append(z.T.dot(W).dot(z))
    return pls, disaggs, z, W

#####################################################
# REDDIT
#####################################################

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

######################################################################
# run NA game for various values of lambda

lam_list = [0.1, 0.6,0.7,0.8,0.9,1] #let's try this for now
res_dict={}
res2_dict={}

max_iter=10
gam=0.2 # arbitrary

for lam in lam_list:
    pls, disaggs, z, W = am(A,s,lam,reduce_pls=False,gam=0,max_iters=max_iter)
    pls2, disaggs2, z2, W2 = am(A,s,lam,reduce_pls=True,gam=gam,max_iters=max_iter)
    
    res_dict[lam] = (pls,disaggs,z,W)
    res2_dict[lam] = (pls2,disaggs2,z2,W2)

with open('reddit_res_large_lambda.pkl', 'wb') as f:
    pickle.dump([res_dict, res2_dict], f)

#####################################################
# READ TWITTER
#####################################################

# # load adjacency matrix
# n_twitter=548
# A = np.zeros([n_twitter, n_twitter])
# z_dict={i:[] for i in range(n_twitter)}

# with open('Twitter/edges_twitter.txt','r') as f:
#     reader=csv.reader(f,delimiter='\t')
#     for u,v in reader:
#         A[int(u)-1,int(v)-1] = 1
#         A[int(v)-1,int(u)-1] = 1

# # load opinions
# with open('Twitter/twitter_opinion.txt','r') as f:
#     reader=csv.reader(f,delimiter='\t')
#     for u,v,w in reader:
#         z_dict[int(u)-1].append(float(w))

# # remove nodes not ocnnected in graph
# not_connected = np.argwhere(np.sum(A,0)==0)
# A=np.delete(A,not_connected,axis=0)
# A=np.delete(A,not_connected,axis=1)
# n_twitter = n_twitter-len(not_connected)

# # create z (avg-ing posts)
# z = [np.mean(z_dict[i]) for i in range(n_twitter)]
# z=np.array(z)

# # create initial opinions from z
# L = np.diag(np.sum(A,0)) - A
# s = (L+np.eye(n_twitter)).dot(z)
# s=np.minimum(np.maximum(s,0),1)

# ###################################
# # run NA game for various values of lambda

# lam_list = [0,0.1,0.2,0.3,0.4] #let's try this for now
# res_dict={}
# res2_dict={}

# max_iter=10
# gam=0.2 # arbitrary

# for lam in lam_list:
#     pls, disaggs, z, W = am(A,s,lam,reduce_pls=False,gam=0,max_iters=max_iter)
#     pls2, disaggs2, z2, W2 = am(A,s,lam,reduce_pls=True,gam=gam,max_iters=max_iter)
    
#     res_dict[lam] = (pls,disaggs,z,W)
#     res2_dict[lam] = (pls2,disaggs2,z2,W2)

# with open('twitter_res.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
#     pickle.dump([res_dict, res2_dict], f)