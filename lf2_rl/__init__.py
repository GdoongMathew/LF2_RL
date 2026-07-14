"""LF2 RL training code — distributed APPO/IMPALA via Ray + RLlib.

This package is intentionally importable on both the DGX learner (Linux,
pure modules only) and the Windows rollout nodes (where the sub-packages
that touch ``lf2_gym.windows`` are loaded by worker processes via Ray's
``runtime_env.worker_process_setup_hook``).
"""
