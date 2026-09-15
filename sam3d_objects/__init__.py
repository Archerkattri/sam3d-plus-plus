# Copyright (c) Meta Platforms, Inc. and affiliates.
import os

# Allow skipping initialization for lightweight tools
if not os.environ.get('LIDRA_SKIP_INIT'):
    try:
        import sam3d_objects.init
    except ModuleNotFoundError as exc:
        # This source-only snapshot does not include Meta's optional upstream
        # environment/bootstrap module. Keep solver imports usable while still
        # surfacing unrelated missing dependencies.
        if exc.name != "sam3d_objects.init":
            raise
        import warnings

        warnings.warn(
            "sam3d_objects.init is absent; running the source-only acceleration "
            "surface without upstream environment initialization",
            RuntimeWarning,
            stacklevel=2,
        )
