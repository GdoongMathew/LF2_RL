"""RLlib env-creator registration for the LF2 parallel env.

Design goals:

* **Safe to import from either side** — the DGX Linux learner AND the
  Windows rollout worker can both `import lf2_rl.rllib.worker_register`
  without the import cascade dragging in ``win32``/``pymem``/``pyautogui``.
  We achieve this by keeping the Windows game bindings (`lf2_gym.windows.*`)
  *inside* the creator function's body, so the module load itself only
  touches cross-platform code.

* **Register once, propagate everywhere** — :func:`setup` calls Ray Tune's
  ``register_env`` which stores the creator inside a shared Ray actor. All
  connected workers see the registration automatically; no per-worker
  runtime-env setup hook needed. This is what makes the module usable
  from both the ``.api_stack(enable_env_runner_and_connector_v2=True)``
  (new API) and the old-stack ``RolloutWorker`` code path (which does
  *not* reliably honour ``runtime_env.worker_process_setup_hook``).

Import order:

* Driver (train.py) — `import lf2_rl.rllib.worker_register`; the bottom
  of this file self-calls :func:`setup` so registration happens as a
  side effect.

* Workers — nothing to do. Ray syncs the registration for them.
"""

from __future__ import annotations

from typing import Any

from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env

from lf2_rl.rllib.spec import LF2_PARALLEL_ENV, ChannelsLastWrapper


def _make_env(config: dict[str, Any] | None = None):
    """RLlib env creator — instantiates :class:`Lf2ParallelEnv` on a Windows
    worker. Only invoked on the rollout side; a Linux driver hits this
    module during ``register_env`` but never *calls* this function.
    """
    # Lazy import: keeps ``lf2_gym.windows.*`` out of the driver's import
    # graph while still making the creator picklable and callable on
    # Windows rollout workers.
    from lf2_gym.windows.parallel_env import Lf2ParallelEnv

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
        "num_players",
        "native_window_height",
        "native_window_width",
    ):
        cfg.pop(driver_only_key, None)
    env = Lf2ParallelEnv(**cfg)
    if channels_last:
        env = ChannelsLastWrapper(env)
    return ParallelPettingZooEnv(env)


def setup() -> None:
    """Idempotent one-shot registration of the LF2 env creator.

    Safe to call multiple times — ``register_env`` overwrites in place.
    """
    register_env(LF2_PARALLEL_ENV, _make_env)


# Importing this module is *itself* the registration — call
# :func:`setup` as a module-level side effect so ``import
# lf2_rl.rllib.worker_register`` on the driver is enough to make
# ``env="lf2_parallel"`` resolve on every worker in the cluster.
setup()
