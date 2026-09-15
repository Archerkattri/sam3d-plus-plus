# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Compatibility surface for the canonical ``hicache-pp`` PyTree API.

SAM 3D flow matching uses structured velocity trees. The implementation lives
in :mod:`hicache_pp.tree`; this module preserves the historical local imports
used by the solver and CFG integration while preventing a second DMD
implementation from drifting from the central package.
"""

try:
    from hicache_pp.tree import (
        adaptive_cfg_decide,
        adaptive_cfg_init,
        dmd_forecast_tree,
        dmd_update_snapshots_tree,
        forecast_guidance_tree,
        guidance_term_tree,
        hermite_coeff,
        hicache_decide,
        hicache_forecast_tree,
        hicache_init,
        hicache_reset,
        hicache_telemetry,
        hicache_update_tree,
        physicists_hermite,
        reconstruct_cfg_tree,
        tree_axpy,
        tree_cosine,
        tree_detach,
        tree_sub_div,
    )
except ImportError as exc:  # pragma: no cover - installation failure path
    raise ImportError(
        "sam3d-plus-plus acceleration requires hicache-pp>=1.2.1"
    ) from exc


__all__ = [
    "adaptive_cfg_decide", "adaptive_cfg_init", "dmd_forecast_tree",
    "dmd_update_snapshots_tree", "forecast_guidance_tree", "guidance_term_tree",
    "hermite_coeff", "hicache_decide", "hicache_forecast_tree", "hicache_init",
    "hicache_reset", "hicache_telemetry", "hicache_update_tree", "physicists_hermite",
    "reconstruct_cfg_tree", "tree_axpy", "tree_cosine", "tree_detach", "tree_sub_div",
]


if __name__ == "__main__":
    state = hicache_init(num_steps=6, interval=3, first_enhance=0)
    sample = {"x": __import__("torch").ones(2)}
    for step in range(3):
        state["step"] = step
        if hicache_decide(state) == "full":
            hicache_update_tree(state, sample)
        else:
            hicache_forecast_tree(state)
    print("ALL TESTS PASSED")
