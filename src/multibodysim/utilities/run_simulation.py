import numpy as np
import orjson, json
from pathlib import Path
from multibodysim import FlexibleNonSymmetricSimulator


config = orjson.loads(Path(
    "/Users/jalalelhazzat/Documents/Packages/configuration/flexible/Input-Shaping/bus_1_mode_1_ZV_Shaping_conf.json"
).read_bytes())
sim = FlexibleNonSymmetricSimulator(config)

theta0            = config["q_initial"]["q3"]
theta_target      = theta0 + np.deg2rad(30.0)
theta_dot_target  = 0.0

Kp = 5
Kd = 14

sim.set_attitude_manoeuver(theta_target, theta_dot_target, Kp, Kd)

sim.run_simulation()
