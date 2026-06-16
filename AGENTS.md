# Agent Notes

This repository is research code for a thesis simulator. Be careful, work in
small increments, and prefer changes that preserve the user's ability to inspect
the physics step by step.

## Current Priorities

- Preserve the single-angle flexible model; it remains useful and should not be
  casually refactored while the multi-angle model is being developed.
- Treat the multi-angle model as the active experimental path for distributed
  torque allocation.
- Prefer British English in new variable, method, and documentation names.
- Keep configuration vocabulary consistent: use `parameters`,
  `q_central_angle`, `u_central_angle`, `q_relative_angle_*`, and
  `u_relative_angle_*`.

## Control/Optimisation Workflow

For new control, allocation, or optimisation ideas:
- Do not implement a large feature in one patch.
- First agree the mathematical object and assumptions with the user.
- If the method introduces unfamiliar maths, pause for a worked toy example
  before coding.
- Build in small slices: math helper, toy test, unit test, simulator
  integration, notebook validation.
- After each slice, stop for review before adding the next conceptual layer.
- Distinguish clearly between "the code solves the stated problem" and "the
  control design is physically good."

## Working Style

- Make small, reviewable patches.
- Do not run the full test suite by default; it is slow. Run targeted tests that
  cover the files changed.
- Do not silently change physics assumptions. If a change affects modelling,
  explain the mathematical reason before or alongside the code.
- Keep notebook-facing helpers lightweight and avoid putting notebook display
  concerns directly into core dynamics modules.
