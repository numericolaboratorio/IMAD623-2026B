"""
oc_poisson.py
"""
import numpy as np
import scipy.sparse as sp
from pathlib import Path
import matplotlib.pyplot as plt

def slap(N):
    """Matriz de Laplace discreta en el cuadrado unitario con N nodos internos por dirección."""
    h = 1/(N+1) #paso de la malla
    diagonales = [4*np.ones(N), -1*np.ones(N-1),-1*np.ones(N-1)]
    D = sp.diags(diagonales,[0,-1,1], format='csr')
    A = sp.kron(sp.eye(N, format ='csr'),D) + sp.diags([-1*np.ones(N**2-N),-1*np.ones(N**2-N)],[-N,N],format='csr' )
    return A/(h**2)

def int_om(N):
    """Matriz de integración sobre el cuadrado unitario con N nodos internos por dirección."""  
    h = 1/(N+1)
    M = sp.diags(h**2*np.ones(N**2),0,format='csr')
    return M

def evalf(N):
    """Evalúa f(x, y) = x + y en los nodos internos del cuadrado unitario y devuelve el vector en R^{N^2}."""
    f=np.zeros([N,N])
    for i in range(0,N):
        for j in range(0,N):
            f[i,j]=j+(N)*i
            
    return f

def malla(N):
    """Nodos internos del cuadrado unitario, N por dirección."""
    h = 1.0 / (N + 1)
    x = np.linspace(h, 1 - h, N)
    X, Y = np.meshgrid(x, x, indexing="ij")   # X[i, j] = x_i, Y[i, j] = y_j
    return X, Y, h

def eval(f, N):

    X, Y, _ = malla(N)  
    return np.asarray(f(X, Y)).reshape(N * N, 1)


def showsurf(u,N, title=""):
    """Muestra la superficie u(x, y) en los nodos internos del cuadrado unitario."""
    X, Y, _ = malla(N)
    if isinstance(u, tuple):
        u = u[0]
    values = np.asarray(u, dtype=float)
    if values.size != N * N:
        raise ValueError(f"u debe contener N*N={N * N} valores; contiene {values.size}.")
    Z = values.reshape(N, N)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="viridis")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("u")
    ax.set_title(title)
    plt.show()

