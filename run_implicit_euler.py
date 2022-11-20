import numpy as np
import matplotlib.pyplot as plt
from assign import *
from q7_params import *
from visualisation import *
import matplotlib.animation as pyanim


def run_implicit_euler():
    '''Runs implicit_function based on number of maximum iterations specified'''
    gr = grid_points(Lx, Ly, T, P, Q, N)

    u0mat = u0(gr.x, gr.y, height, centre, radius)

    print("Choose Maxits: ")

    maxits = int(input())

    U, iterations = implicit_Euler(gr, kappa, alpha, Fisher, u0mat, tol, maxits)

    myanim = animate_soln(gr, U)

    writervideo = pyanim.FFMpegWriter(fps=20)

    if(maxits == 0):
        myanim.save('IMEX.mp4', writer=writervideo)
    else:
        myanim.save('implicit_euler.mp4', writer=writervideo)

run_implicit_euler()

