from assign import *
from visualisation import *
import matplotlib.animation as pyanim
from scipy.linalg import norm


def convergence(): # I wasn't sure if to use the implicit euler with maxits > 0.
    P = 64
    Q = 64
    T = 1
    maxits = 0 # I have left it at 0 here so the solution completes in under a minute
    NExact = 1024
    rows = 4

    grExact = grid_points(Lx, Ly, T, P, Q, NExact)

    u0mat = u0(grExact.x, grExact.y, height, centre, radius)

    UExact, iterExact = implicit_Euler(grExact, kappa, alpha, Fisher, u0mat, tol, maxits)
    preErr = 0.0
    N = NExact >> 5
    M = (P+1) * (Q+1)
    print("   N |  Error    Rate")
    for row in range(rows):
        gr = grid_points(Lx, Ly, T, P, Q, N)
        u0mat = u0(gr.x, gr.y, height, centre, radius)

        U, iters = implicit_Euler(gr, kappa, alpha, Fisher, u0mat, tol, maxits)
        err = norm(U[N,:,:].reshape(M) - UExact[NExact,:,:].reshape(M), np.inf)

        if(row == 0):
            print(f"{N:4d} | {err:6.4f}")
        else:
            rate = np.log2(preErr/err)
            print(f"{N:4d} | {err:6.4f}  {rate:6.4f}")
        preErr = err
        N *= 2

convergence()