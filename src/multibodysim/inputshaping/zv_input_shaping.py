import numpy as np

class InputShaper:
    """
    FIR input shaper: shaped_cmd(t) = sum_i A[i] * base_cmd(t - T[i])
    """
    def __init__(self, A, T):
        self.A = np.array(A, dtype=float)
        self.T = np.array(T, dtype=float)
        assert np.all(self.T >= 0.0)
        
        # normalize weights defensively
        self.A = self.A / np.sum(self.A)

    @staticmethod
    def zvd(omega, zeta):
        if omega is None and zeta is None:
            A = np.array([1.0, 0.0, 0.0], dtype=float)
            T = np.array([0.0, 0.0, 0.0], dtype=float)
        else:
            # omega in rad/s
            zeta = float(zeta)
            omega = float(omega)
            wd = omega * np.sqrt(max(1.0 - zeta**2, 1e-12))
            Td = 2.0 * np.pi / wd

            K = np.exp(-zeta * np.pi / np.sqrt(max(1.0 - zeta**2, 1e-12)))

            A = np.array([1.0, 2.0*K, K**2], dtype=float)
            T = np.array([0.0, 0.5*Td, 1.0*Td], dtype=float)
            A = A / np.sum(A)

        return InputShaper(A, T)

    def shape(self, t, raw_command):
        # raw_command: callable(time) -> value
        y = 0.0
        for Ai, Ti in zip(self.A, self.T):
            y += Ai * raw_command(t - Ti)
        return y
