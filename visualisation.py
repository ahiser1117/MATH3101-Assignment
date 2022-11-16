import matplotlib.pyplot as plt
import matplotlib.animation as pyanim
from numpy import meshgrid

def animate_soln(gr, U, interval=200):
    x, y, t = gr.x, gr.y, gr.t
    X, Y = meshgrid(x, y)
    zmax = U.max()
    fig = plt.figure(1)
    ax = fig.add_subplot(projection='3d')
    Z = U[0,:,:].T
    ax.plot_wireframe(X, Y, Z)
    ax.set_zlim(0, 1.2*zmax)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u')
    ax.set_title(f't = {0:4.2f}')
    def animate(n):
        ax.cla()
        Z = U[n,:,:].T
        ax.plot_wireframe(X, Y, Z)
        ax.set_zlim(0, 1.2*zmax)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('u')
        ax.set_title(f't = {t[n]:4.2f}')
    N = len(t) - 1
    myanim = pyanim.FuncAnimation(fig, animate, frames=N+1, interval=interval)
    plt.show()
    return myanim

def snapshot(gr, Un, figno=1):
    x, y = gr.x, gr.y
    X, Y = meshgrid(x, y)
    zmax = Un.max()
    fig = plt.figure(figno)
    ax = fig.add_subplot(projection='3d')
    Z = Un.T
    ax.plot_wireframe(X, Y, Z)
    ax.set_zlim(0, 1.2*zmax)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u')
    plt.show()
    return fig



