from .base import ControlOutput


class PlanarAttitudeController:
    def __init__(self, plant_view):
        self.plant_view = plant_view

    def reset(self):
        pass

    def compute(self, t, q, u, Md=None):
        return ControlOutput(tau_ff=0.0, tau_fb=0.0)
