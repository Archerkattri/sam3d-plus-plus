# Copyright (c) Meta Platforms, Inc. and affiliates.
import optree
import torch
import time
from functools import partial

from sam3d_objects.data.utils import tree_tensor_map
from .accel import (
    hicache_init,
    hicache_decide,
    hicache_update_tree,
    hicache_forecast_tree,
    hicache_telemetry,
    tree_detach,
    dmd_update_snapshots_tree,
    dmd_forecast_tree,
)
from hicache_pp import CacheBudget, CacheBudgetRuntime, RunIdentity, stable_digest


def linear_approximation_step(x_t, dt, velocity):
    # x_tp1 = x_t + velocity * dt
    x_tp1 = tree_tensor_map(lambda x, v: x + v * dt, x_t, velocity)
    return x_tp1


def gradient(output, x, create_graph: bool = False):
    tensors, pyspec = optree.tree_flatten(
        x, is_leaf=lambda x: isinstance(x, torch.Tensor)
    )
    grad_outputs = [torch.ones_like(output).detach() for _ in tensors]
    grads = torch.autograd.grad(
        output,
        tensors,
        grad_outputs=grad_outputs,
        create_graph=create_graph,
    )
    return optree.tree_unflatten(pyspec, grads)


class ODESolver:
    def enable_hicache(self, interval: int = 4, max_order: int = 1, first_enhance: int = 2,
                       end_enhance=None, sigma: float = 0.5, budget=None,
                       max_horizon=None, max_memory_mb=None, audit_budget: int = 0):
        """Enable HiCache (Hermite velocity forecast) — EULER solver only. Forecast the
        (CFG-combined) velocity tree on skipped steps instead of calling dynamics_fn,
        skipping (interval-1)/interval of the model evaluations. Training-free; native
        (the solver calls the accel helpers directly — no monkey-patching)."""
        self._hicache_cfg = dict(interval=interval, max_order=max_order,
                                 first_enhance=first_enhance, end_enhance=end_enhance, sigma=sigma)
        self._hicache_budget = budget or CacheBudget(
            backend="hermite", allowed_stages=("flow",),
            max_horizon=max(1, interval - 1) if max_horizon is None else max_horizon,
            quality_preset="adapter-default", max_memory_mb=max_memory_mb,
            audit_budget=audit_budget, fallback="full",
        )
        return self

    def enable_dmd(self, interval: int = 4, first_enhance: int = 2, end_enhance=None,
                   history: int = 5, max_order: int = 2, sigma: float = 0.5, budget=None,
                   max_horizon=None, max_memory_mb=None, audit_budget: int = 0):
        """Enable the DMD/Prony EXPONENTIAL velocity forecaster (exponential analogue of
        HiCache) on the Euler solver — forecasts skipped-step velocity trees from raw
        snapshots via fractional eigenvalue powers; Hermite covers warm-up. See accel.py."""
        self._hicache_cfg = dict(interval=interval, max_order=max_order, first_enhance=first_enhance,
                                 end_enhance=end_enhance, sigma=sigma, backend="dmd", history=history)
        self._hicache_budget = budget or CacheBudget(
            backend="dmd", allowed_stages=("flow",),
            max_horizon=max(1, interval - 1) if max_horizon is None else max_horizon,
            quality_preset="adapter-default", max_memory_mb=max_memory_mb,
            audit_budget=audit_budget, fallback="full",
        )
        return self

    def disable_hicache(self):
        self._hicache_cfg = None
        self._hicache_budget = None
        return self

    def get_hicache_telemetry(self):
        """Return telemetry from the active or most recently completed trajectory."""
        if getattr(self, "_hicache", None) is not None:
            return hicache_telemetry(self._hicache)
        return getattr(self, "_last_hicache_telemetry", None)

    def get_hicache_manifest(self):
        """Return the latest identity-bound budget manifest, if available."""
        runtime = getattr(self, "_hicache_runtime", None)
        if runtime is not None:
            return runtime.manifest.as_dict()
        return getattr(self, "_last_hicache_manifest", None)

    @staticmethod
    def _tree_signature(value):
        leaves = [leaf for leaf in optree.tree_flatten(
            value, is_leaf=lambda item: isinstance(item, torch.Tensor)
        )[0] if isinstance(leaf, torch.Tensor)]
        shapes = [tuple(int(dim) for dim in leaf.shape) for leaf in leaves]
        first = leaves[0] if leaves else None
        return shapes, (str(first.dtype) if first is not None else "unknown"), (str(first.device) if first is not None else "unknown")

    def _make_hicache_runtime(self, state, x_init, times):
        shapes, dtype, device = self._tree_signature(x_init)
        identity = RunIdentity(
            model_id=type(self).__name__,
            run_id=str(state["run_id"]),
            schedule_digest=stable_digest([float(value) for value in times]),
            cfg_branch="cfg-combined",
            conditioning_id="cfg-combined",
            stage="flow",
            token_layout_digest=stable_digest({"shapes": shapes}),
            dtype=dtype,
            device=device,
            batch_id=stable_digest({"shapes": shapes, "dtype": dtype, "device": device}),
        )
        return CacheBudgetRuntime(
            self._hicache_budget, identity,
            source_digest=stable_digest({"adapter": "sam3d-plus-plus", "contract": "budget-v1"}),
            config_digest=self._hicache_budget.digest,
        )

    @staticmethod
    def _tree_memory_mb(value):
        leaves = [leaf for leaf in optree.tree_flatten(
            value, is_leaf=lambda item: isinstance(item, torch.Tensor)
        )[0] if isinstance(leaf, torch.Tensor)]
        for leaf in leaves:
            if leaf.is_cuda:
                return float(torch.cuda.memory_allocated(leaf.device)) / (1024.0 * 1024.0)
        return None

    def step(self, dynamics_fn, x_t, t, dt, *args, **kwargs):
        raise NotImplementedError

    def solve_iter(self, dynamics_fn, x_init, times, *args, **kwargs):
        # HiCache state is per-trajectory and only valid for a pure Euler solver (exactly
        # one dynamics_fn evaluation per step). Reset it at the start of every run.
        cfg = getattr(self, "_hicache_cfg", None)
        self._hicache = (hicache_init(num_steps=len(times) - 1, **cfg)
                         if cfg is not None and type(self) is Euler else None)
        self._hicache_runtime = (
            self._make_hicache_runtime(self._hicache, x_init, times)
            if self._hicache is not None else None
        )
        self._last_hicache_telemetry = None
        self._last_hicache_manifest = None
        x_t = x_init
        if cfg is not None and type(self) is not Euler:
            self._last_hicache_telemetry = {
                "status": "unsupported_solver",
                "solver": type(self).__name__,
                "decisions": {"full": 0, "forecast": 0},
                "method_counts": {},
                "fallbacks": {"unsupported_solver": 1},
            }
        try:
            for t0, t1 in zip(times[:-1], times[1:]):
                dt = t1 - t0
                x_t = self.step(dynamics_fn, x_t, t0, dt, *args, **kwargs)
                yield x_t, t0
        finally:
            if self._hicache is not None:
                self._last_hicache_telemetry = hicache_telemetry(self._hicache)
            if self._hicache_runtime is not None:
                self._last_hicache_manifest = self._hicache_runtime.manifest.as_dict()
            self._hicache = None
            self._hicache_runtime = None

    def solve(self, dynamics_fn, x_init, times, *args, **kwargs):
        for x_t, _ in self.solve_iter(dynamics_fn, x_init, times, *args, **kwargs):
            pass
        return x_t


# https://en.wikipedia.org/wiki/Euler_method
class Euler(ODESolver):
    def step(self, dynamics_fn, x_t, t, dt, *args, **kwargs):
        hc = getattr(self, "_hicache", None)
        budget_runtime = getattr(self, "_hicache_runtime", None)
        timer = time.perf_counter()
        decision = hicache_decide(hc) if hc is not None else "full"
        budget_decision = None
        if hc is not None and budget_runtime is not None:
            budget_decision = budget_runtime.decide(
                "flow",
                horizon=int(hc.get("counter", 0)) if decision == "forecast" else 0,
                method=hc.get("backend", "hermite"),
                supported=True,
                memory_mb=self._tree_memory_mb(x_t),
            )
            if budget_decision.mode == "fallback" and budget_decision.method == "full" and decision == "forecast":
                hc["type"] = "full"
                hc["counter"] = 0
                hc["activated_steps"].append(hc["step"])
                decisions = hc["telemetry"]["decisions"]
                decisions["forecast"] = max(0, int(decisions.get("forecast", 0)) - 1)
                decisions["full"] = int(decisions.get("full", 0)) + 1
                decision = "full"
        if decision == "forecast":
            # Forecast the velocity tree from cached anchors; skip dynamics_fn. The
            # exponential (DMD) backend forecasts from raw velocity snapshots; the default
            # Hermite backend from finite-difference derivatives.
            if hc.get("backend") == "dmd":
                velocity = dmd_forecast_tree(hc)
            else:
                velocity = hicache_forecast_tree(hc)
            hc["step"] += 1
            if budget_runtime is not None:
                budget_runtime.record_measurement(
                    "flow",
                    budget_decision.method if budget_decision is not None else hc.get("backend", "hermite"),
                    wall_time_ms=(time.perf_counter() - timer) * 1000.0,
                )
        else:
            velocity = dynamics_fn(x_t, t, *args, **kwargs)
            if hc is not None:
                vdet = tree_detach(velocity)
                hicache_update_tree(hc, vdet)
                if hc.get("backend") == "dmd":
                    dmd_update_snapshots_tree(hc, vdet, hc["history"])
                hc["step"] += 1
            if budget_runtime is not None:
                budget_runtime.record_measurement(
                    "flow", "full", wall_time_ms=(time.perf_counter() - timer) * 1000.0,
                )
        x_tp1 = linear_approximation_step(x_t, dt, velocity)
        return x_tp1


# https://arxiv.org/abs/2505.05470
class SDE(ODESolver):
    def __init__(self, **kwargs):
        super().__init__()
        self.sde_strength = kwargs.get("sde_strength", 0.1)

    def step(self, dynamics_fn, x_t, t, dt, *args, **kwargs):
        velocity = dynamics_fn(x_t, t, *args, **kwargs)
        sigma = 1 - t
        var_t = sigma / (1 - torch.tensor(sigma).clamp(min=dt))
        std_dev_t = (
            torch.sqrt(variance) * self.sde_strength
        )  # self.sde_strength = alpha

        def compute_mean(x, v):
            drift_term = x * (std_dev_t**2 / (2 * sigma) * dt)
            velocity_term = v * (1 + std_dev_t**2 * (1 - sigma) / (2 * sigma)) * dt
            return x + drift_term + velocity_term

        prev_sample_mean = tree_tensor_map(compute_mean, x_t, velocity)

        # Generate noise and compute final sample using tree_tensor_map
        def add_noise(mean_val):
            variance_noise = torch.randn_like(mean_val)
            return mean_val + std_dev_t * torch.sqrt(torch.tensor(dt)) * variance_noise

        prev_sample = tree_tensor_map(add_noise, prev_sample_mean)

        return prev_sample


# https://en.wikipedia.org/wiki/Midpoint_method
class Midpoint(ODESolver):
    def step(self, dynamics_fn, x_t, t, dt, *args, **kwargs):
        half_dt = 0.5 * dt

        x_mid = Euler.step(self, dynamics_fn, x_t, t, half_dt, *args, **kwargs)

        velocity_mid = dynamics_fn(x_mid, t + half_dt, *args, **kwargs)
        x_tp1 = linear_approximation_step(x_t, dt, velocity_mid)
        return x_tp1


# https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
class RungeKutta4(ODESolver):

    def k1(self, dynamics_fn, x_t, t, dt, *args, **kwargs):
        return dynamics_fn(x_t, t, *args, **kwargs)

    def k2(self, dynamics_fn, x_t, t, dt, k1, *args, **kwargs):
        x_k1 = linear_approximation_step(x_t, dt * 0.5, k1)
        return dynamics_fn(x_k1, t + dt * 0.5, *args, **kwargs)

    def k3(self, dynamics_fn, x_t, t, dt, k2, *args, **kwargs):
        x_k2 = linear_approximation_step(x_t, dt * 0.5, k2)
        return dynamics_fn(x_k2, t + dt * 0.5, *args, **kwargs)

    def k4(self, dynamics_fn, x_t, t, dt, k3, *args, **kwargs):
        x_k3 = linear_approximation_step(x_t, dt, k3)
        return dynamics_fn(x_k3, t + dt, *args, **kwargs)

    def step(self, dynamics_fn, x_t, t, dt, *args, **kwargs):
        k1 = self.k1(dynamics_fn, x_t, t, dt, *args, **kwargs)
        k2 = self.k2(dynamics_fn, x_t, t, dt, k1, *args, **kwargs)
        k3 = self.k3(dynamics_fn, x_t, t, dt, k2, *args, **kwargs)
        k4 = self.k4(dynamics_fn, x_t, t, dt, k3, *args, **kwargs)

        def compute_velocity(k1, k2, k3, k4):
            return (k1 + 2 * k2 + 2 * k3 + k4) / 6

        velocity_k = tree_tensor_map(compute_velocity, k1, k2, k3, k4)
        x_tp1 = linear_approximation_step(x_t, dt, velocity_k)
        return x_tp1
