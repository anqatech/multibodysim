import numpy as np
import orjson, json
from pathlib import Path
from multibodysim import FlexibleNonSymmetricSimulator


config = orjson.loads(Path(
    "/Users/jalalelhazzat/Documents/Packages/configuration/flexible/Keplerian-Orbit/kepler_orbit_mode_1_bodies_3_conf.json"
).read_bytes())
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

sim.set_nadir_pointing(Kp=Kp, Kd=Kd)
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------

# # ----------------------------------------------------------------------
# # -------------------------- Input Shaping Setup --------------------------
# # ----------------------------------------------------------------------

# zeta  = 0.012
# omega = 18.35

# omega_d = omega * np.sqrt(1 - zeta**2)
# Tr = 3 * (2 * np.pi / omega_d) 


# # ----------------------------------------------------------------------
# # ----------------------------------------------------------------------

# sim.set_attitude_manoeuver(theta_target, theta_dot_target, Kp, Kd, omega, zeta, Tr, shaping_flag=False)

results = sim.run_simulation(eval_flag=False)
