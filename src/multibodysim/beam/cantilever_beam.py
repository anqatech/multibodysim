import numpy as np
import sympy as sm
from scipy.integrate import trapezoid
from scipy.optimize import root_scalar


class CantileverBeam:
    def __init__(self, length, E, I, n):
        self.L = length
        self.E = E
        self.I = I

        self.nb_modes = n

        self.betas = self._generate_betas(n=self.nb_modes)
        self.sigmas = self._generate_sigmas()

    def mode_shape_symbolic(self, s, mode):
        arg = self.betas[mode - 1] * s / self.L
        phi = sm.cosh(arg) - sm.cos(arg) - self.sigmas[mode - 1] * ( sm.sinh(arg) - sm.sin(arg) )
        return phi

    def mode_shape(self, s, mode):
        arg = self.betas[mode - 1] * s / self.L
        phi = (np.cosh(arg) - np.cos(arg) - self.sigmas[mode - 1] * (np.sinh(arg) - np.sin(arg)))
        return phi
    
    def mode_shape_mean(self, n_points=200):
        s_vals = np.linspace(0, self.L, n_points)
        phi_vals = self.mode_shape(s_vals, 1)
        return trapezoid(phi_vals, s_vals) / self.L
    
    def modal_stiffness(self, mode):
        # Create symbolic mode shape using beam's own parameters
        s = sm.Symbol('s')
        arg = self.betas[mode - 1] * s / self.L
        phi = (sm.cosh(arg) - sm.cos(arg) - self.sigmas[mode - 1] * (sm.sinh(arg) - sm.sin(arg)))
        
        # Calculate modal stiffness
        phi_dd = phi.diff(s, 2)
        k_modal = sm.integrate(self.E * self.I * (phi_dd)**2, (s, 0, self.L))
        return k_modal
    
    def _generate_betas(self, n=5):
        def f(b): return np.cosh(b) * np.cos(b) + 1.0
        betas = []
        k = 0
        eps = 1e-12
        while len(betas) < n:
            a = k * np.pi + eps
            b = (k + 1.0) * np.pi - eps
            sol = root_scalar(f, bracket=(a, b), method="brentq", xtol=eps, rtol=eps, maxiter=200)
            betas.append(sol.root)
            k += 1
        return np.array(betas)

    def _generate_sigmas(self):
        num = np.cosh(self.betas) + np.cos(self.betas)
        den = np.sinh(self.betas) + np.sin(self.betas)
        return num / den
