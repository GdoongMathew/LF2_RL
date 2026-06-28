"""Distributed async RL training entrypoint for LF2 (APPO / IMPALA).

Architecture:

* **Driver / learner** (this script) runs on the DGX Linux node. It only
  imports :mod:`lf2_rl.rllib.spec` — a pure module that declares the env
  name, obs/action spaces and policy-mapping. *Nothing* on this side ever
  reaches into ``lf2_gym.windows`` or any ``win32`` / ``pymem`` /
  ``pyautogui`` library.
* **Rollout EnvRunner workers** run on Windows nodes hosting LF2 windows.
  Each worker process loads :mod:`lf2_rl.rllib.worker_register` via Ray's
  ``runtime_env.worker_process_setup_hook``; that module does
  ``register_env`` so the worker can resolve ``env="lf2_parallel"`` to a
  real :class:`Lf2ParallelEnv`.

Usage (single machine, smoke test)::

    python -m lf2_rl.rllib.train --num-env-runners 1 --mode mix

Usage (cluster): start the Ray head on the DGX and Windows workers first
(see ``lf2_rl/rllib/cluster/``), then run this on the head with
``--address auto`` and ``--num-env-runners`` equal to the number of LF2
windows.
"""

from __future__ import annotations

import argparse


def build_config(args: argparse.Namespace):
    """Build the APPO/IMPALA AlgorithmConfig for LF2 multi-agent training."""
    from ray.rllib.algorithms.appo import APPOConfig
    from ray.rllib.algorithms.impala import IMPALAConfig

    from lf2_rl.rllib.spec import (
        LF2_PARALLEL_ENV,
        SHARED_POLICY_ID,
        build_spaces_from_config,
        policy_mapping_fn,
    )

    env_config: dict = {
        "windows_name": args.windows_name,
        "player_ids": tuple(args.player_ids),
        "mode": args.mode,
        "frame_stack": args.frame_stack,
        "frame_skip": args.frame_skip,
        "gray_scale": args.gray_scale,
        "reset_skip_sec": args.reset_skip_sec,
        "downscale": args.downscale,
        "channels_last": True,
    }
    if args.character:
        env_config["character"] = args.character
    if args.num_actions:
        env_config["num_actions"] = args.num_actions
    if args.img_h:
        env_config["img_h"] = args.img_h
    if args.img_w:
        env_config["img_w"] = args.img_w

    # Declare obs/action spaces purely from config so RLlib never has to
    # construct an env on the driver to probe them.
    obs_space, act_space = build_spaces_from_config(env_config, channels_last=True)

    config_cls = IMPALAConfig if args.algo == "impala" else APPOConfig
    config = (
        config_cls()
        .environment(
            env=LF2_PARALLEL_ENV,
            env_config=env_config,
            observation_space=obs_space,
            action_space=act_space,
            disable_env_checking=True,
        )
        .framework("torch")
        # Async rollout workers (the LF2 "actors"). Place them on Windows nodes
        # that advertise the custom `LF2_WINDOW` resource (see cluster scripts).
        .env_runners(
            num_env_runners=args.num_env_runners,
            num_envs_per_env_runner=1,
            rollout_fragment_length=args.rollout_fragment_length,
        )
        # The learner trains on the DGX GPU.
        .learners(
            num_learners=args.num_learners,
            num_gpus_per_learner=args.num_gpus_per_learner,
        )
        .training(
            train_batch_size=args.train_batch_size,
            gamma=args.gamma,
            lr=args.lr,
        )
        # One shared policy for all 4 in-window agents. Spaces are passed in
        # explicitly so neither the driver nor the policy setup needs to
        # build an env.
        .multi_agent(
            policies={SHARED_POLICY_ID: (None, obs_space, act_space, {})},
            policy_mapping_fn=policy_mapping_fn,
        )
    )

    if args.lf2_window_resource:
        # Pin rollout EnvRunners to Windows nodes that run LF2.
        config = config.env_runners(
            num_cpus_per_env_runner=1,
            custom_resources_per_env_runner={"LF2_WINDOW": 1},
        )

    return config, LF2_PARALLEL_ENV


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LF2 distributed async RL (APPO/IMPALA).")
    p.add_argument("--algo", choices=["appo", "impala"], default="appo")
    p.add_argument("--address", default=None, help="Ray cluster address, e.g. 'auto'.")
    p.add_argument("--windows-name", default="Little Fighter 2")
    p.add_argument("--player-ids", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--mode", choices=["mix", "picture", "info"], default="mix")
    p.add_argument("--frame-stack", type=int, default=4)
    p.add_argument("--frame-skip", type=int, default=1)
    p.add_argument("--downscale", type=int, default=2)
    p.add_argument("--gray-scale", action="store_true", default=True)
    p.add_argument("--no-gray-scale", dest="gray_scale", action="store_false")
    p.add_argument("--reset-skip-sec", type=int, default=2)

    # Space-declaration knobs (the learner needs these so it never has to
    # build a live env just to learn its obs/action shape).
    p.add_argument(
        "--character",
        default=None,
        help="Character name to derive the action space from "
        "(default: Template = 9 basic moves).",
    )
    p.add_argument(
        "--num-actions",
        type=int,
        default=None,
        help="Explicit discrete action count (overrides --character).",
    )
    p.add_argument(
        "--img-h",
        type=int,
        default=None,
        help="Override the observation image height; default = derived from "
        "the standard 800x600 LF2 window + --downscale.",
    )
    p.add_argument("--img-w", type=int, default=None, help="Override observation image width.")

    p.add_argument("--num-env-runners", type=int, default=1)
    p.add_argument("--num-learners", type=int, default=1)
    p.add_argument("--num-gpus-per-learner", type=float, default=1.0)
    p.add_argument("--lf2-window-resource", action="store_true", default=False)

    p.add_argument("--rollout-fragment-length", type=int, default=50)
    p.add_argument("--train-batch-size", type=int, default=500)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--lr", type=float, default=5e-4)

    p.add_argument("--stop-timesteps", type=int, default=6_000_000)
    p.add_argument("--checkpoint-dir", default="LF2_RL_Model")
    p.add_argument("--checkpoint-freq", type=int, default=50)
    return p.parse_args(argv)


def main(argv=None) -> None:
    import ray

    args = parse_args(argv)
    # Every Ray worker process imports ``lf2_rl.rllib.worker_register`` at
    # startup. That module is Windows-only — it pulls in
    # ``lf2_gym.windows.parallel_env`` at top level and runs
    # ``register_env`` as a side effect, so workers can resolve the
    # ``"lf2_parallel"`` env name without the learner ever loading any
    # Windows code.
    ray.init(
        address=args.address,
        runtime_env={
            "worker_process_setup_hook": "lf2_rl.rllib.worker_register:setup",
        },
    )

    config, _ = build_config(args)
    algo = config.build()

    iteration = 0
    try:
        while True:
            result = algo.train()
            iteration += 1
            ts = result.get("num_env_steps_sampled_lifetime", 0)
            reward = result.get("env_runners", {}).get("episode_return_mean")
            print(f"[iter {iteration}] timesteps={ts} episode_return_mean={reward}")

            if args.checkpoint_freq and iteration % args.checkpoint_freq == 0:
                path = algo.save(args.checkpoint_dir)
                print(f"  checkpoint -> {path}")

            if ts >= args.stop_timesteps:
                break
    finally:
        algo.save(args.checkpoint_dir)
        algo.stop()
        ray.shutdown()


if __name__ == "__main__":
    main()
