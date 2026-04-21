import numpy as np
import orjson, json
from pathlib import Path
from multibodysim import FlexibleNonSymmetricSimulator
from multibodysim import PlanarAttitudeController


conf_directory  = "/Users/jalalelhazzat/Documents/Packages/configuration/flexible/PD-Control/"
conf_filename   = "kepler_orbit_mode_1_bodies_3_conf.json"

config_path = Path(conf_directory) / conf_filename
with open(config_path, 'rb') as f:
    config = orjson.loads(f.read())

sim = FlexibleNonSymmetricSimulator(config)

# ----------------------------------------------------------------------
# -------------------------- PD Control Setup --------------------------
# ----------------------------------------------------------------------

zeta_cl  = 0.707
omega_cl = 0.01
J_eff    = 152.5

Kp = J_eff * omega_cl**2
Kd = 2 * zeta_cl * J_eff * omega_cl

print(f"Kp = {Kp:.1f}, Kd = {Kd:.1f}\n")

# # ----------------------------------------------------------------------
# # ------------------------- Input Shaping Setup ------------------------
# # ----------------------------------------------------------------------

zeta  = 0.012
omega = 18.35

omega_d = omega * np.sqrt(1 - zeta**2)
Tr = 3 * (2 * np.pi / omega_d) 

print(f"Input Shaping Zeta = {zeta:.6f}, Omega = {omega:.4f}, Omega_d = {omega_d:.4f}, Tr = {Tr:.3f}\n")

# ----------------------------------------------------------------------
# ---------------------------- Angle Target ----------------------------
# ----------------------------------------------------------------------

theta0            = config["q_initial"]["q3"]
theta_target      = theta0 + np.deg2rad(30.0)
theta_dot_target  = 0.0

# # ----------------------------------------------------------------------
# # ----------------------------------------------------------------------

ctrl = PlanarAttitudeController(sim.plant_view)
ctrl.configure_attitude_pd(
    theta_target=theta_target,
    theta_dot_target=theta_dot_target,
    Kp=Kp,
    Kd=Kd,
    Tr=Tr,
    use_input_shaping=False,
    shaper=None,
    omega=omega,
    zeta=zeta,
)
sim.set_controller(ctrl)

# # ----------------------------------------------------------------------
# # ----------------------------------------------------------------------

results = sim.run_simulation(eval_flag=False)
