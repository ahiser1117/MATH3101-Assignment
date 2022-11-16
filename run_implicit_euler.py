import numpy as np
import matplotlib.pyplot as plt
from assign import *
from q7_params import *
from visualisation import *
import matplotlib.animation as pyanim

gr = grid_points(Lx, Ly, T, P, Q, N)

u0mat = u0(gr.x, gr.y, height, centre, radius)

U = implicit_Euler(gr, kappa, alpha, Fisher, u0mat, None, None)

myanim = animate_soln(gr, U)

writervideo = pyanim.FFMpegWriter(fps=10)
myanim.save('fisher.mp4', writer=writervideo)



