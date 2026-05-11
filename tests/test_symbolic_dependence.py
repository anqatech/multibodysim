from __future__ import annotations

from dataclasses import dataclass

import sympy as sm
import sympy.physics.mechanics as me

from multibodysim.analysis.symbolic_dependence import symbolic_dependence_report


@dataclass
class FakeDynamics:
    frames: dict[str, me.ReferenceFrame]
    q: sm.Matrix
    qd: sm.Matrix
    u: sm.Matrix
    ud: sm.Matrix
    parameter_symbols: dict[str, sm.Symbol]
    bus_torque_symbols: dict[str, sm.Symbol]


def fake_dynamics():
    inertial_frame = me.ReferenceFrame("N")
    t = me.dynamicsymbols._t
    q1, q2 = me.dynamicsymbols("q1 q2")
    u1, u2 = me.dynamicsymbols("u1 u2")
    D, m_bus_1 = sm.symbols("D m_bus_1")
    tau_1 = sm.symbols("tau_1")

    return FakeDynamics(
        frames={"inertial": inertial_frame},
        q=sm.Matrix([q1, q2]),
        qd=sm.Matrix([q1.diff(t), q2.diff(t)]),
        u=sm.Matrix([u1, u2]),
        ud=sm.Matrix([u1.diff(t), u2.diff(t)]),
        parameter_symbols={"D": D, "m_bus_1": m_bus_1},
        bus_torque_symbols={"bus_1": tau_1},
    )


def test_symbolic_dependence_report_classifies_scalar_dependencies():
    dynamics = fake_dynamics()
    q1 = dynamics.q[0]
    u2 = dynamics.u[1]
    q1d = dynamics.qd[0]
    u2d = dynamics.ud[1]
    D = dynamics.parameter_symbols["D"]
    tau_1 = dynamics.bus_torque_symbols["bus_1"]
    alpha = sm.symbols("alpha")

    report = symbolic_dependence_report(q1 + u2 + q1d + u2d + D + tau_1 + alpha, dynamics)

    assert report["q"] == [q1]
    assert report["u"] == [u2]
    assert report["qd"] == [q1d]
    assert report["ud"] == [u2d]
    assert report["parameters"] == [D]
    assert report["torques"] == [tau_1]
    assert alpha in report["other_symbols"]


def test_symbolic_dependence_report_accepts_vectors_and_lists():
    dynamics = fake_dynamics()
    frame = dynamics.frames["inertial"]
    q2 = dynamics.q[1]
    u1 = dynamics.u[0]
    D = dynamics.parameter_symbols["D"]
    tau_1 = dynamics.bus_torque_symbols["bus_1"]

    vector = (q2 + D) * frame.x + (u1 + tau_1) * frame.y
    report = symbolic_dependence_report([vector, D * tau_1], dynamics)

    assert report["q"] == [q2]
    assert report["u"] == [u1]
    assert report["parameters"] == [D]
    assert report["torques"] == [tau_1]
