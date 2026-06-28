"""Windows-only RLlib env-creator registration.

Loaded once per Ray worker process via the ``runtime_env`` setup hook in
:mod:`lf2_rl.rllib.train` — i.e. only on the LF2-running Windows nodes.
Top-level imports therefore freely reach into ``lf2_gym.windows`` without
worrying about the learner side.

The setup is a *one-shot side effect* that the rollout workers need before
they create their env via the RLlib registry name ``lf2_parallel``.
"""

from __future__ import annotations

from typing import Any

from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env

from lf2_gym.windows.parallel_env import Lf2ParallelEnv
from lf2_rl.rllib.spec import LF2_PARALLEL_ENV, ChannelsLastWrapper


def _make_env(config: dict[str, Any] | None = None):
    """RLlib env creator — only ever called on Windows env-runner workers."""
    cfg = dict(config or {})
    channels_last = cfg.pop("channels_last", True)
    # Driver-only knobs that the worker shouldn't try to forward into the
    # underlying ``Lf2ParallelEnv`` constructor.
    for driver_only_key in (
        "num_actions",
        "character",
        "img_h",
        "img_w",
        "mp_max",
        "hp_max",
        "native_window_height",
        "native_window_width",
    ):
        cfg.pop(driver_only_key, None)
    env = Lf2ParallelEnv(**cfg)
    if channels_last:
        env = ChannelsLastWrapper(env)
    return ParallelPettingZooEnv(env)


def setup() -> None:
    """Idempotent one-shot registration of the LF2 env creator with Ray Tune."""
    register_env(LF2_PARALLEL_ENV, _make_env)


# Importing this module is *itself* the registration — useful when a script
# does ``import lf2_rl.rllib.worker_register`` instead of relying on the
# Ray setup hook.
setup()
