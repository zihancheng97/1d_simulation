import numpy as np

def apply_1gate(psi, i, op):
    psi = psi.reshape(2**i, 2, -1)
    psi = np.einsum('ab, ibj -> iaj', op, psi).reshape(-1)
    return psi

def apply_2gate(psi, i, j, op):
    if i<j:
        psi = psi.reshape(2**i, 2, 2**(j-i-1), 2, -1)
        psi = np.einsum('abcd, icjdk -> iajbk',op.reshape(2, 2, 2, 2), psi).reshape(-1)
    elif i>j:
        psi = psi.reshape(2**j, 2, 2**(i-j-1), 2, -1)
        psi = np.einsum('abcd, idjck -> ibjak',op.reshape(2, 2, 2, 2), psi).reshape(-1)
    return psi

def apply_3gate(psi, i, j, k, op):
    if i<j<k:
        psi = psi.reshape(2**i, 2, 2**(j-i-1), 2, 2**(k-j-1), 2, -1)
        psi = np.einsum('abcdef, idjekfl -> iajbkcl',op.reshape(2, 2, 2, 2, 2, 2), psi).reshape(-1)
    elif k<i<j:#L-2, L-1, 0
        psi = psi.reshape(2**k, 2, 2**(i-k-1), 2, 2**(j-i-1), 2, -1)
        psi = np.einsum('abcdef, ifjdkel -> icjakbl',op.reshape(2, 2, 2, 2, 2, 2), psi).reshape(-1)
    elif j<k<i:#L-1, 0, 1
        psi = psi.reshape(2**j, 2, 2**(k-j-1), 2, 2**(i-k-1), 2, -1)
        psi = np.einsum('abcdef, iejfkdl -> ibjckal',op.reshape(2, 2, 2, 2, 2, 2), psi).reshape(-1)
    return psi

def measure(psi, i):
    X = [[0.,1.],[1.,0.]]
    dim_l = 2**i 
    psi = psi.reshape(dim_l, 2, -1)
    prob0 = np.linalg.norm(psi[:,0])**2 
    if np.random.rand() < prob0:
        psi[:,1] = 0. 
    else:
        psi[:,0] = 0.
        psi = np.einsum('ij,ajb->aib', X, psi)
    psi = psi.ravel() 
    return psi/np.linalg.norm(psi)