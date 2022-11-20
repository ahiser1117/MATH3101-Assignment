import numpy as np
from scipy.sparse.linalg import spsolve
from dataclasses import dataclass
from scipy.sparse import spdiags, kron, eye
from scipy import sparse
from q7_params import *
import time

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
    return alpha * (u - np.multiply(u, u))

def dF(u, alpha):
    return np.diag(alpha * (1 - 2 * u))

def implicit_Euler(gr, kappa, alpha, reaction, u0mat, tol, maxits):
    # Read in arguments from grid points
    x = gr.x
    y = gr.y
    t = gr.t
    P = x.size-1
    Q = y.size-1
    N = t.size-1
    dt = gr.dt

    # Allocate memory and read in initial values
    U = np.empty((N+1, P+1, Q+1))
    iterations = np.zeros((N+1))
    A = Poisson_matrix(gr, kappa)
    M = (P + 1) * (Q + 1)
    U[0, :, :] = u0mat[:, :]
    
    # 0th iteration steps
    fV, dfV = reaction(U[0,:,:].reshape(M), alpha)
    V = u0mat[:, :].reshape(M)
    for n in range(1, N+1):
        # Equation (9) to solve for V
        V = spsolve(eye(M) + dt * A, V + dt * fV)
        
        # fV and dfV values for following step
        fV, dfV = reaction(V, alpha)
        
        dVmax = tol + 1
        j = 0
        while(j < maxits and dVmax > tol):
            # Calculate LHS and RHS of equation (10)        
            a = (eye(M) + dt * A - dt * dfV)
            b = (U[n-1,:,:].reshape(M) + dt * fV - (eye(M) + dt * A) @ V)
            
            # Solve for dV
            dV = spsolve(a, b)
            V = V + dV
            
            dVmax = np.abs(dV.max())
            j += 1
            fV, dfV = reaction(V, alpha)

        iterations[n] = j
        
        # Copy in calculated values to output matrix
        U[n, :, :] = V.reshape(P+1,Q+1)

    return U, iterations

def u0(x, y, H, C, R):
    x, y = np.meshgrid(x, y)
    return H * np.exp(-((x - C[0])**2 + (y - C[1])**2)/R**2)






