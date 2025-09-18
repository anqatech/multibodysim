import numpy as np
import sympy as sm
from scipy.integrate import trapezoid
from scipy.optimize import root_scalar


class CantileverBeam:
    def __init__(self, length, E, I, beta1, sigma1, s):
        self.L = length
        self.E = E
        self.I = I
        self.beta1 = beta1
        self.sigma1 = sigma1
        self.s = s
        
    def mode_shape(self, s):
        arg = self.beta1 * s / self.L
        phi1 = (np.cosh(arg) - np.cos(arg) - self.sigma1 * (np.sinh(arg) - np.sin(arg)))
        return phi1
    
    def mode_shape_mean(self, n_points=200):
        s_vals = np.linspace(0, self.L, n_points)
        phi1_vals = self.mode_shape(s_vals)
        return trapezoid(phi1_vals, s_vals) / self.L
    
    def modal_stiffness_symbolic(self):
        # Create symbolic mode shape using beam's own parameters
        arg = self.beta1 * self.s / self.L
        phi1 = (sm.cosh(arg) - sm.cos(arg) - 
                self.sigma1 * (sm.sinh(arg) - sm.sin(arg)))
        
        # Calculate modal stiffness
        phi1_dd = phi1.diff(self.s, 2)
        k_modal = sm.integrate(self.E * self.I * (phi1_dd)**2, (self.s, 0, self.L))
        return k_modal

    def betas_cantilever(n=5):
        def f(b): return np.cosh(b) * np.cos(b) + 1.0
        betas = []
        k = 0
        eps = 1e-12
        while len(betas) < n:
            a, b = k * np.pi + eps, (k + 1) * np.pi - eps
            sol = root_scalar(f, bracket=(a, b), method="brentq", xtol=eps, rtol=eps, maxiter=200)
            betas.append(sol.root)
            k += 1
        return np.array(betas)
