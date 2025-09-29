import numpy as np
import scipy.linalg as la

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

def measure_noreset(psi, i):
    dim_l = 2**i 
    psi = psi.reshape(dim_l, 2, -1)
    prob0 = np.linalg.norm(psi[:,0])**2 
    if np.random.rand() < prob0:
        psi[:,1] = 0. 
    else:
        psi[:,0] = 0.
    psi = psi.ravel() 
    return psi/np.linalg.norm(psi)

def measure_prob(psi, i):
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
    return psi/np.linalg.norm(psi), prob0


def calc_SA(psi, i0, l0, N, ns):
    v = psi.reshape(2 ** i0, 2 ** l0, 2 ** (N - i0 - l0))
    rho = np.tensordot(v.conj(), v, axes = [[0, 2], [0, 2]]).reshape(2 ** (l0),  2 ** (l0)) # i0, N - i0 - l0, i0, N - i0 - l0
    eigs = np.real_if_close(la.eigvals(rho))
    eigs = np.maximum(eigs, 0.)
    #eigs /= eigs.sum()    
    Ss = []
    for n in ns:
        if np.isclose(n, 1):
            S = -np.sum(eigs * np.log(eigs + 1e-20))
        else:
            S_n = np.sum(eigs ** n)
            S = -1. / (n - 1) * np.log(S_n + 1e-20)
        Ss.append(S)
    return np.array(Ss)

def calc_SAB(psi, i0, l0, i1, l1, N, ns):
    v = psi.reshape(2 ** i0, 2 ** l0, 2 ** (i1 - l0 - i0), 2** (l1), 2**(N - l1 - i1))
    rho = np.tensordot(v.conj(), v, axes = [[0, 2, 4], [0, 2, 4]]).reshape(2 ** (l0 + l1),  2 ** (l0 + l1)) # i0, N - i0 - l0, i0, N - i0 - l0
    eigs = np.real_if_close(la.eigvals(rho))
    eigs = np.maximum(eigs, 0.)
    #eigs /= eigs.sum()    
    Ss = []
    for n in ns:
        if np.isclose(n, 1):
            S = -np.sum(eigs * np.log(eigs + 1e-20))
        else:
            S_n = np.sum(eigs ** n)
            S = -1. / (n - 1) * np.log(S_n + 1e-20)
        Ss.append(S)
    return np.array(Ss)


def calc_I3(psi, N, ns):
    SA = calc_SA(psi, 0, N // 4, N, ns)
    SB = calc_SA(psi, N // 4, N // 4, N, ns)
    SC = calc_SA(psi, 2 * (N // 4), N // 4, N, ns)
    SAB = calc_SA(psi, 0, 2*(N // 4), N, ns)
    SBC = calc_SA(psi, N // 4,2 * (N // 4), N, ns)
    SAC = calc_SAB(psi, 0, N // 4, 2 * (N // 4), N // 4, N, ns)
    SABC = calc_SA(psi, 3 * (N // 4), N-3*(N // 4), N, ns)
    I3 = np.real_if_close(SA + SB + SC + SABC - SAB - SAC - SBC)
    return I3