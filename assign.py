import numpy as np
from dataclasses import dataclass
from scipy.sparse import spdiags, kron
from q7_params import *


@dataclass
class grid:
    x: np.array([])
    y: np.array([])
    t: np.array([])
    dx: float
    dy: float
    dt: float

def grid_points(Lx, Ly, T, P, Q, N):
    x = np.linspace(0, Lx, P+1)
    dx = Lx/P
    y = np.linspace(0, Ly, Q+1)
    dy = Ly/Q
    t = np.linspace(0, T, N+1)
    dt = T/N
    gr = grid(x, y, t, dx, dy, dt)
    return gr


def d2matrix(P, Dx, kappa):
    data = np.array([np.full((P+1), -1), np.full((P+1), 2), np.full((P+1), -1)])
    diags = np.array([-1, 0, 1])
    Ax = spdiags(data, diags, P+1, P+1).toarray()
    Ax[0, 1] = -2
    Ax[P, P-1] = -2
    Ax = (kappa / Dx**2) * Ax
    return Ax

def Poisson_matrix(gr, kappa):
    P = gr.x.size-1
    Q = gr.y.size-1
    Ax = d2matrix(P, gr.dx, kappa)
    Ay = d2matrix(Q, gr.dy, kappa)
    Ix = np.identity(P+1)
    Iy = np.identity(Q+1)
    A = kron(Ax, Iy) + kron(Ix, Ay)
    return A

def Fisher(V, alpha):
    f = F(V, alpha)
    df = dF(V, alpha)
    return f, df


def F(u, alpha):
    f = np.empty((u.size))
    for i in range(u.size):
        f[i] = alpha * u[i] * (1 - u[i])
    return f

def dF(u, alpha):
    df = np.empty((u.size))
    for i in range(u.size):
        df[i] = alpha * (1 - 2 * u[i])
    return df

def implicit_Euler(gr, kappa, alpha, reaction, u0mat, tol, maxits):
    x = gr.x
    y = gr.y
    t = gr.t
    P = x.size-1
    Q = y.size-1
    N = t.size-1
    dt = gr.dt

    U = np.empty((N+1, P+1, Q+1))
    A = Poisson_matrix(gr, kappa)
    U[0, :, :] = u0mat[:, :]
    V = U[0,:,:]
    M = (P + 1) * (Q + 1)

    for n in range(1, N+1):
        print(n)
        fV, dfV = reaction(V.reshape((M,1)), alpha)
        # print(fV)
        # print(f"fV: {fV.shape}, V: {V.reshape(M).shape}, A: {A.shape}, M: {M}")
        V = (V.reshape(M) + dt * fV) * np.linalg.inv(np.identity(M) + dt * A)
        # print(f"fV: {fV.shape}, V: {V.reshape(M).shape}, A: {A.shape}, M: {M}")
        U[n, :, :] = V.reshape(P+1,Q+1)
        # print(V.max())

    return U

def u0(x, y, H, C, R):
    u0 = np.empty((x.size, y.size))
    for i in range(x.size):
        for j in range(y.size):
            u0[i, j] = H * np.exp(-((x[i] - C[0])**2 + (y[j] - C[1])**2)/R**2)
    return u0






