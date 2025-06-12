import numpy as np
from scipy.linalg import qr

def random_2_cl():
    """Generate random 2-qubit Cliffords."""
    # [A. D. Corcoles, Supplementary material for `Process verification of two-qubit quantum gates by randomized benchmarking']
    Ha = np.array([[1., 1.], [1., -1.]])/np.sqrt(2)
    S = np.array([[1., 0.], [0., 1j]])
    sx = np.array([[0., 1.], [1., 0.]])
    sy = np.array([[0., -1j], [1j, 0]])
    sz = np.array([[1., 0.], [0., -1]])
    Rs = np.cos(np.pi/3)*np.eye(2)-1j*np.sin(np.pi/3)*(sx+sy+sz)/np.sqrt(3)
    A = [np.eye(2), Ha, S, Ha@S, S@Ha, Ha@S@Ha]
    B = [np.eye(2), sx, sy, sz]
    C = [np.eye(2), Rs, Rs@Rs]
    CNOT = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 0., 1.], [0., 0., 1., 0.]])
    iSWAP = np.array([[1., 0., 0., 0.], [0., 0., 1j, 0.], [0., 1j, 0., 0.], [0., 0., 0., 1.]])
    SWAP = np.array([[1., 0., 0., 0.], [0., 0., 1., 0.], [0., 1., 0., 0.], [0., 0., 0., 1.]])

    ia1 = np.random.randint(0, 6)
    ib1 = np.random.randint(0, 4)
    C1 = A[ia1]@B[ib1]
    ia2 = np.random.randint(0, 6)
    ib2 = np.random.randint(0, 4)
    C2 = A[ia2]@B[ib2]

    is1 = np.random.randint(0, 3)
    S1 = C[is1]
    is2 = np.random.randint(0, 3)
    S2 = C[is2]

    x = np.random.random()

    if x < 0.05:
        Cl = np.kron(C1, C2)
    elif 0.05 < x < 0.5:
        Cl = np.kron(C1, C2)
        S_prod = np.kron(S1, S2)
        Cl = S_prod@CNOT@Cl
    elif 0.5 < x < 0.95:
        Cl = np.kron(C1, C2)
        S_prod = np.kron(S1, S2)
        Cl = S_prod@iSWAP@Cl
    else:
        Cl = np.kron(C1, C2)
        Cl = SWAP@Cl   

    return Cl    


def qr_haar(N):
    """Generate a Haar-random matrix using the QR decomposition."""
    # Step 1
    A, B = np.random.normal(size=(N, N)), np.random.normal(size=(N, N))
    Z = A + 1j * B

    # Step 2
    Q, R = qr(Z)

    # Step 3
    Lambda = np.diag([R[i, i] / np.abs(R[i, i]) for i in range(N)])

    # Step 4
    return np.dot(Q, Lambda)


def qr_haar_sp(N):
    #Convention: I_n x I_2 
    """
    Sample a Haar-random unitary symplectic matrix from USp(d/2).
    Requires d to be even.
    """
    # [Francesco Mezzadri, How to generate random matrices from the classical compact groups]
    if N % 2 != 0:
        raise ValueError("d must be even for USp(d/2)")

    n = N // 2

    E0 = np.array([[1., 0.], [0., 1.]])
    E1 = 1j*np.array([[1., 0.], [0., -1.]])
    E2 = np.array([[0., 1.], [-1., 0.]])
    E3 = 1j*np.array([[0., 1.], [1., 0.]])
    # Omega = np.kron(np.eye(n), E2)

    # Step 1: Generate a random quaternionic matrix (as 4 real matrices)
    A = np.random.randn(n, n)
    B = np.random.randn(n, n)
    C = np.random.randn(n, n)
    D = np.random.randn(n, n)

    # Step 2: Embed quaternionic matrix into complex representation
    # Q = A + Bi + Cj + Dk = E0xA + E1xB + E2xC + E3xD --> 2n x 2n complex matrix
    # Using complex representation of quaternions:
    # q = a + b*i + c*j + d*k -> [[a + bi, c + di], [-c + di, a - bi]]

    # M = np.kron(A, E0) + np.kron(B, E1) + np.kron(C, E2) + np.kron(D, E3)

    # Step 3: QR decomposition

    Q = np.eye(2*n)
    for k in range(n):
        #2*(n-k) x 2
        v = np.kron(A[k:, k:k+1], E0) + np.kron(B[k:, k:k+1], E1) + np.kron(C[k:, k:k+1], E2) + np.kron(D[k:, k:k+1], E3)
        v_norm= np.sqrt(np.trace(v.conj().T@v)/2)
        v0 = v/v_norm
        x1 = v[:2, :2]#E0*A[k,k] + E1*B[k,k] + E2*C[k,k] + E3*D[k,k]
        x1_norm = np.sqrt(np.trace(x1@x1.conj().T)/2)
        q = x1/x1_norm
        u = v0.copy()
        u[:2, :2] += q
        u_norm= np.sqrt(np.trace(u@u.conj().T)/2)
        u0 = u/u_norm
        uuT = u0@u0.conj().T
        Hk = -np.kron(np.eye(n-k), q.conj().T)@(np.eye(2*n-2*k)-2*uuT)
        Hk_t = np.eye(2*n, dtype=complex)
        Hk_t[2*k:, 2*k:] = Hk.copy()
        Q = Q@Hk_t.conj().T
    return Q