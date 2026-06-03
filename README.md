# multibodysim

`multibodysim` is a Python package for symbolic and numerical simulation of
planar spacecraft multibody dynamics, with particular focus on flexible
appendages, gravity-gradient effects, and attitude-control studies.

The package is currently developed as a research simulator. Its main use case is
to build equations of motion symbolically with SymPy Mechanics, convert those
equations to numerical functions, and integrate the resulting system with SciPy.
The code is designed around chain-like spacecraft made from rigid buses and
flexible panels, for example:

```text
panel -- bus -- panel -- bus -- panel
```

or longer distributed structures such as:

```text
panel -- bus -- panel -- bus -- panel -- bus -- panel
```

## What The Package Models

The active development path focuses on two flexible spacecraft models.

The single-angle flexible model assumes all rigid bodies share one attitude
degree of freedom. This is useful for studying flexible-body dynamics,
gravity-gradient response, and centralised attitude control when the whole
spacecraft rotates as one planar structure.

The multi-angle flexible model extends this by giving each rigid bus its own
attitude coordinate. The central bus has an absolute attitude,
`q_central_angle`, while the other buses use relative attitude coordinates such
as `q_relative_angle_bus_1`. This model is intended for studying distributed
torque allocation across multiple buses.

Both models support flexible panel modal coordinates, configurable gravity
gradient effects, Keplerian orbital motion, static bus torque inputs, and
optional feedback control.

## Package Layout

```text
src/multibodysim/
  analysis/       metrics and symbolic-dependence utilities
  beam/           beam mode-shape helpers
  controllers/    attitude PD and nadir-pointing controllers
  flexible/       single-angle flexible spacecraft dynamics and simulator
  inputshaping/   ZV/ZVD input-shaping utilities
  multiangle/     multi-angle flexible spacecraft dynamics and simulator
  plotting/       notebook-oriented plotting helpers
  rigid/          older rigid-body symbolic and simulation models
  utilities/      small simulation helpers
```

## How A Simulation Works

A simulation is driven by a configuration dictionary. The configuration defines:

- the spacecraft graph and central body;
- rigid and flexible body names and types;
- physical parameters such as masses, bus size, panel length, stiffness, and
  orbital parameters;
- initial coordinates and speeds;
- external forces, bus torques, and torque allocation weights;
- integration options such as method, relative tolerance, and per-state absolute
  tolerances.

The simulator then follows this workflow:

1. Build a symbolic dynamics object from the configuration.
2. Define generalized coordinates, generalized speeds, frames, points,
   velocities, accelerations, partial velocities, active forces, and inertia
   forces.
3. Derive Kane equations in the form
   `mass_matrix * ud = forcing`.
4. For the multi-angle simulator, load or generate autowrap evaluators for
   runtime use.
5. Build initial conditions from the configuration.
6. Integrate the first-order state system with `scipy.integrate.solve_ivp`.
7. Return a results dictionary containing time histories, solver diagnostics,
   controller torques, centre-of-mass diagnostics, and named state variables.

## Installation

From the repository root:

```bash
conda activate avril-2026
python -m pip install -e ".[dev]"
```

The `dev` extra installs the test and notebook dependencies used during current
development, including `pytest`, `pytest-cov`, `jupyterlab`, and `pandas`.

## Minimal Usage

Single-angle flexible simulator:

```python
from multibodysim.flexible.flexible_ns_simulator import (
    FlexibleNonSymmetricSimulator,
)

simulator = FlexibleNonSymmetricSimulator(config)
results = simulator.run_simulation(eval_flag=True)
```

Multi-angle flexible simulator:

```python
from multibodysim.multiangle import MultiAngleFlexibleSimulator

simulator = MultiAngleFlexibleSimulator(config)
results = simulator.run_simulation(eval_flag=True)
```

Multi-angle simulator construction prepares the autowrap evaluators
automatically. On a fresh cache, this can compile local generated artifacts
before the first simulation run.

Autowrap evaluators can be validated explicitly from notebooks when needed:

```python
from multibodysim.codegen import validate_autowrap_evaluators

validation_report = validate_autowrap_evaluators(simulator.dynamics)
```

Attitude PD control:

```python
from multibodysim.controllers.pd_attitude import PlanarAttitudeController

controller = PlanarAttitudeController(simulator.plant_view)
controller.configure_attitude_pd(
    theta_target=theta_target,
    theta_dot_target=0.0,
    Kp=Kp,
    Kd=Kd,
    Tr=Tr,
    use_input_shaping=False,
)

simulator.set_controller(controller)
results = simulator.run_simulation(eval_flag=True)
```

## Analysis And Plotting

The `analysis` package provides small metric helpers for comparing simulations,
controllers, and torque allocation strategies. These helpers intentionally return
plain dictionaries or display-ready rows so notebooks can decide how to present
the data.

The `plotting` package contains notebook-oriented Matplotlib functions for
state, speed, flexible-mode, torque, and nadir-angle diagnostics.

## Development Status

This package is research code under active development. The single-angle
flexible model is the more established path. The multi-angle model is being
developed to support distributed bus attitude dynamics and torque allocation
studies. Public APIs may still change as the physics model and notebook workflow
settle.
