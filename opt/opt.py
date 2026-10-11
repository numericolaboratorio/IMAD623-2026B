"""
opt.py
"""
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

np.set_printoptions(precision=6, suppress=False)

class opt_unconstrained:
    # Problema sin restricciones min f(x) con información de primer orden.

    def __init__(self, f, grad, x0, nombre=""):
        self._f = f
        self._grad = grad
        self.x0 = np.asarray(x0, dtype=float)
        self.n = self.x0.size
        self.nombre = nombre
        self.reset()

    def reset(self):
        self.nf = 0   # evaluaciones de f
        self.ng = 0   # evaluaciones de grad f

    def f(self, x):
        self.nf += 1
        return self._f(x)

    def grad(self, x):
        self.ng += 1
        return self._grad(x)


def verificar_gradiente(f, grad, x, h=1e-6):
    g_fd = np.array([(f(x + h * e) - f(x - h * e)) / (2 * h) for e in np.eye(x.size)])
    g = grad(x)
    return np.linalg.norm(g - g_fd) / max(1.0, np.linalg.norm(g))

