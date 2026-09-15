"""CPU contract tests for the stock SAM 3D PyTree integration."""

import hicache_pp.tree as central
import torch

from sam3d_objects.model.backbone.generator.flow_matching import accel
from sam3d_objects.model.backbone.generator.flow_matching.solver import Euler, RungeKutta4


def _tree(value):
    return {"x": torch.tensor([value, value + 1.0]), "nested": {"y": torch.tensor([[value]])}}


def test_accel_facade_matches_central_pytree_api():
    assert accel.hicache_init is central.hicache_init
    assert accel.dmd_forecast_tree is central.dmd_forecast_tree
    assert accel.hicache_telemetry is central.hicache_telemetry


def test_dmd_fit_reuse_and_nonuniform_fallback(monkeypatch):
    state = central.hicache_init(num_steps=12, interval=1, first_enhance=0,
                                 backend="dmd", history=5)
    for step in range(4):
        state["step"] = step
        state["activated_steps"].append(step)
        value = _tree(float(step))
        central.hicache_update_tree(state, value)
        central.dmd_update_snapshots_tree(state, value)

    calls = 0
    original = central._dmd_fit_flat

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(central, "_dmd_fit_flat", counted)
    state["step"] = 5
    central.dmd_forecast_tree(state)
    central.dmd_forecast_tree(state)
    assert calls == 1

    nonuniform = central.hicache_init(num_steps=10, interval=1, backend="dmd")
    for step in (0, 1, 3, 4):
        nonuniform["step"] = step
        nonuniform["activated_steps"].append(step)
        value = _tree(float(step))
        central.hicache_update_tree(nonuniform, value)
        central.dmd_update_snapshots_tree(nonuniform, value)
    nonuniform["step"] = 5
    central.dmd_forecast_tree(nonuniform)
    assert nonuniform["telemetry"]["fallbacks"]["dmd_nonuniform_or_short_tail"] == 1


def test_solver_telemetry_reset_disable_and_non_euler_boundary():
    calls = 0

    def dynamics(x, _t):
        nonlocal calls
        calls += 1
        return {"x": torch.ones_like(x["x"]), "nested": {"y": torch.ones_like(x["nested"]["y"])}}

    solver = Euler().enable_dmd(interval=3, first_enhance=0, history=5)
    list(solver.solve_iter(dynamics, _tree(0.0), torch.linspace(0, 1, 10)))
    first = solver.get_hicache_telemetry()
    assert calls < 9
    assert first["decisions"]["forecast"] > 0
    assert first["decisions"]["full"] + first["decisions"]["forecast"] == 9

    calls = 0
    list(solver.solve_iter(dynamics, _tree(0.0), torch.linspace(0, 1, 10)))
    second = solver.get_hicache_telemetry()
    assert calls < 9
    assert second["decisions"] == first["decisions"]

    solver.disable_hicache()
    calls = 0
    list(solver.solve_iter(dynamics, _tree(0.0), torch.linspace(0, 1, 4)))
    assert calls == 3
    assert solver.get_hicache_telemetry() is None

    calls = 0
    stock_solver = RungeKutta4().enable_dmd(interval=3)
    list(stock_solver.solve_iter(dynamics, _tree(0.0), torch.linspace(0, 1, 4)))
    assert calls == 12
    assert stock_solver.get_hicache_telemetry()["fallbacks"] == {"unsupported_solver": 1}
