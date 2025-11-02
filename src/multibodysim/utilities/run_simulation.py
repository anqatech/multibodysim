from multibodysim import FlexibleNonSymmetricSimulator
import orjson, json
from pathlib import Path

cfg = orjson.loads(Path(
    "/Users/jalalelhazzat/Documents/Packages/configuration/flexible/flexible_ns_N_bodies_conf.json"
).read_bytes())
sim = FlexibleNonSymmetricSimulator(cfg)
sim.run_simulation()
