"""RLlib integration for LF2 distributed async RL.

This subpackage wires the in-window multi-agent :class:`Lf2ParallelEnv` into
Ray RLlib so that:

* the 4 in-window players are driven by a single **shared policy**, and
* training runs as an **async actor-learner** (APPO / IMPALA): rollout
  ``EnvRunner`` s on Windows machines collect experience while the learner
  trains on the DGX GPU and broadcasts fresh weights back.

These modules import ``ray`` lazily / at call time; install with the ``rl``
extra: ``pip install -e .[rl]``.
"""
