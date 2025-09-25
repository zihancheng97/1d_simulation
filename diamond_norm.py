from scipy import sparse
import cvxpy
import numpy as np
from sympy.combinatorics.named_groups import SymmetricGroup, Permutation


def super_to_choi(op):
    sqrt_shape = int(np.sqrt(op.shape[0]))
    choi = op.reshape([sqrt_shape]*4).transpose(3, 1, 2, 0).reshape(op.shape)
    return choi

def diamond_norm(op, **kwargs):
    def cvx_bmat(mat_r, mat_i):
        """Block matrix for embedding complex matrix in reals"""
        return cvxpy.bmat([[mat_r, -mat_i], [mat_i, mat_r]])
    choi = super_to_choi(op)
    dim_in = int(np.sqrt(op.shape[0]))
    dim_out = int(np.sqrt(op.shape[1]))
    size = dim_in * dim_out

    # SDP Variables to convert to real valued problem
    r0_r = cvxpy.Variable((dim_in, dim_in))
    r0_i = cvxpy.Variable((dim_in, dim_in))
    r0 = cvx_bmat(r0_r, r0_i)

    r1_r = cvxpy.Variable((dim_in, dim_in))
    r1_i = cvxpy.Variable((dim_in, dim_in))
    r1 = cvx_bmat(r1_r, r1_i)

    x_r = cvxpy.Variable((size, size))
    x_i = cvxpy.Variable((size, size))
    iden = sparse.eye(dim_out)

    # Watrous uses row-vec convention for his Choi matrix while we use
    # col-vec. It turns out row-vec convention is requried for CVXPY too
    # since the cvxpy.kron function must have a constant as its first argument.
    c_r = cvxpy.bmat([[cvxpy.kron(iden, r0_r), x_r], [x_r.T, cvxpy.kron(iden, r1_r)]])
    c_i = cvxpy.bmat([[cvxpy.kron(iden, r0_i), x_i], [-x_i.T, cvxpy.kron(iden, r1_i)]])
    c = cvx_bmat(c_r, c_i)

    # Transpose out Choi-matrix to row-vec convention and vectorize.
    choi_vec = np.transpose(
        np.reshape(choi.data, (dim_in, dim_out, dim_in, dim_out)),
        (1, 0, 3, 2)).ravel(order='F')
    choi_vec_r = choi_vec.real
    choi_vec_i = choi_vec.imag

    # Constraints
    cons = [
        r0 >> 0, r0_r == r0_r.T, r0_i == - r0_i.T, cvxpy.trace(r0_r) == 1,
        r1 >> 0, r1_r == r1_r.T, r1_i == - r1_i.T, cvxpy.trace(r1_r) == 1,
        c >> 0
    ]

    # Objective function
    obj = cvxpy.Maximize(choi_vec_r @ cvxpy.vec(x_r) - choi_vec_i @ cvxpy.vec(x_i))
    prob = cvxpy.Problem(obj, cons)
    sol = prob.solve(**kwargs)
    return sol


def proj4():
    G = SymmetricGroup(4)
    elements = list(G.generate_dimino(af=True))
    C1 = {Permutation(3)}
    C2 = {Permutation(0, 1, 2, 3),
            Permutation(0, 1, 3, 2),
            Permutation(0, 2, 1, 3),
            Permutation(0, 2, 3, 1),
            Permutation(0, 3, 1, 2),
            Permutation(0, 3, 2, 1)}
    C3 =  {Permutation(0, 1)(2, 3), Permutation(0, 2)(1, 3), Permutation(0, 3)(1, 2)}
    C4 =  {Permutation(0, 3),
    Permutation(1, 3),
    Permutation(2, 3),
    Permutation(3)(0, 1),
    Permutation(3)(0, 2),
    Permutation(3)(1, 2)}
    C5 =  {Permutation(0, 1, 3),
    Permutation(0, 2, 3),
    Permutation(0, 3, 1),
    Permutation(0, 3, 2),
    Permutation(1, 2, 3),
    Permutation(1, 3, 2),
    Permutation(3)(0, 1, 2),
    Permutation(3)(0, 2, 1)}
    identity = np.kron(np.kron(np.eye(2), np.eye(2)), np.kron(np.eye(2), np.eye(2))).reshape(2, 2, 2, 2, 2, 2, 2, 2)
    proj = np.zeros((256, 256))
    for e1 in elements:
        for e2 in elements:
            p1 = Permutation(e1)
            p2 = Permutation(e2)
            p2_inv = ~Permutation(e2)
            tau = p1*p2_inv
            if tau in C1:
                wg = (17+1./5)/(24**2)
            elif tau in C2:
                wg = (-3+1./5)/(24**2)
            elif tau in C3:
                wg = (5+1./5)/(24**2)
            elif tau in C4:
                wg = (3+1./5)/(24**2)
            elif tau in C5:
                wg = (-4+1./5)/(24**2)
            else:
                print('error')
            
            vec1 = np.moveaxis(identity, [0, 1, 2, 3], p1).reshape(256)
            vec2 = np.moveaxis(identity, [0, 1, 2, 3], p2).reshape(256)
            proj += wg*np.tensordot(vec1, vec2, axes = 0)

    return proj

def proj1():
    bell = np.array([1., 0., 0., 1.])
    proj = np.einsum('i, j -> ij', bell, bell)/2   
    return proj 

def proj2():
    bell = np.array([1., 0., 0., 1.]).reshape(2, 2)
    identity = np.einsum('ij, kl -> ikjl', bell, bell).reshape(16,)
    swap = np.einsum('ij, kl -> iklj', bell, bell).reshape(16,)
    proj = np.einsum('i, j -> ij', identity, identity)/3+\
        np.einsum('i, j -> ij', swap, swap)/3-np.einsum('i, j -> ij', identity, swap)/6-np.einsum('i, j -> ij', swap, identity)/6
    return proj

def proj3():
    bell = np.array([1., 0., 0., 1.]).reshape(2, 2)
    identity = np.einsum('ij, kl -> ikjl', bell, bell).reshape(16,)
    swap = np.einsum('ij, kl -> iklj', bell, bell).reshape(16,)
    identity3 = np.einsum('ikjl, mn -> ikmjln', identity.reshape(2, 2, 2, 2), bell).reshape(64,)
    swap12 = np.einsum('ikjl, mn -> ikmjln', swap.reshape(2, 2, 2, 2), bell).reshape(64,)
    swap13 = np.einsum('ikjl, mn -> imkjnl', swap.reshape(2, 2, 2, 2), bell).reshape(64,)
    swap23 = np.einsum('ikjl, mn -> miknjl', swap.reshape(2, 2, 2, 2), bell).reshape(64,)
    swap123 = np.einsum('ikjl, mn -> ikmnjl', identity.reshape(2, 2, 2, 2), bell).reshape(64,)
    swap321 = np.einsum('ikjl, mn -> ikmlnj', identity.reshape(2, 2, 2, 2), bell).reshape(64,)

    r1 = 1./6*(1./24+2./3)
    r2 =1./6*(1./24)
    r3 = 1./6*(1./24-1./3)
    states = [identity3, swap12, swap13, swap23, swap123, swap321]
    proj = np.zeros((64, 64))
    for i in range(6):
        for j in range(6):
            if i==0:
                if j==0:
                    r = r1
                elif 0<j<4:
                    r = r2
                else:
                    r = r3
            elif 0<i<4:
                if j==0:
                    r = r2
                elif j==i:
                    r = r1
                elif j!=i and 0<j<4:
                    r = r3
                else:
                    r = r2
            elif i>3:
                if j==0:
                    r = r3
                elif 0<j<4:
                    r = r2
                elif j == i:
                    r = r1
                else:
                    r = r3
            proj +=r*np.einsum('i, j -> ij', states[i], states[j])
    return proj

def calc_diamond_norm(Phi, k):
    if k == 1:
        proj = proj1()
    elif k == 2:
        proj = proj2()
    elif k == 3:
        proj = proj3()
    elif k == 4:
        proj = proj4()

    return diamond_norm(Phi-proj)
