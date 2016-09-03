# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 12:04:41 2016

@author: fox
"""
import mdp
import scipy
import numpy as np
from scipy.sparse.linalg import eigs

# Use sparse matrices for generating the weights,
# Note: To compensate this change we modified Oger reservoir_nodes.py to all dot product of sparse matrics with dense matrics
def generate_sparse_w(output_size, specrad, fan_in_res=10, seed=1):
    converged = False
    mdp.numx.random.seed(seed)

    # Initialize reservoir weight matrix from a normal distribution
    nrentries = mdp.numx.int32(output_size * fan_in_res)
    # Keep generating random matrices until convergence
    while not converged:
        try:
            #$%ij = mdp.numx.zeros((2,nrentries))
            ij = mdp.numx.random.randint(0,output_size,(2,nrentries))
            datavec =  mdp.numx.random.randn(nrentries)
            w = scipy.sparse.csc_matrix((datavec, ij),dtype=np.float32, shape=(output_size, output_size))
            we = eigs(w,return_eigenvectors=False,k=3)
            converged = True
            w *= (specrad / mdp.numx.amax(mdp.numx.absolute(we)))
        except:
            pass
    return w.toarray()
    #return w

def generate_sparse_w_in(output_size, input_size, scaling, fan_in_i=2,seed=1):
    import scipy.sparse

    mdp.numx.random.seed(seed)
    # Initialize reservoir weight matrix from a normal distribution
    nrentries = mdp.numx.int32(output_size * fan_in_i)
    # Keep generating random matrices until convergence
    ij = mdp.numx.zeros((2,nrentries))
    ij[0,:] = mdp.numx.random.randint(0,output_size,(1,nrentries))
    ij[1,:] = mdp.numx.random.randint(0,input_size,(1,nrentries))
    datavec =  mdp.numx.random.randn(nrentries)
    w = scaling * scipy.sparse.csc_matrix((datavec, ij),dtype=np.float32, shape=(output_size, input_size))
    return w.toarray()
    #return w