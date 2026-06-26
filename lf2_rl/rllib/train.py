"""Distributed async RL training entrypoint for LF2 (APPO / IMPALA).

Runs a single shared policy over the 4 in-window agents, with rollout
``EnvRunner`` s collecting experience (on Windows machines running LF2) and the
learner training on the DGX GPU. APPO (async PPO) provides the asynchronous
actor->learner->weight-broadcast loop.

Usage (single machine, smoke test)::

    python -m lf2_rl.rllib.train --num-env-runners 1 --mode mix

Usage (cluster): start the Ray head on the DGX and Windows workers first (see
``lf2_rl/rllib/cluster/``), then run this on the head with
``--address auto`` and ``--num-env-runners`` equal to the number of LF2 windows.
"""

from __future__ import annotations

import argparse


def build_config(args: argparse.Namespace):
    """Build the APPO/IMPALA AlgorithmConfig for LF2 multi-agent training."""
    from ray.rllib.algorithms.appo import APPOConfig
    from ray.rllib.algorithms.impala import IMPALAConfig

    from lf2_rl.rllib.env_factory import (
        SHARED_POLICY_ID,
        policy_mapping_fn,
        register,
    )

    env_name = register()

    env_config = {
        "windows_name": args.windows_name,
        "player_ids": tuple(args.player_ids),
        "mode": args.mode,
        "frame_stack": args.frame_stack,
        "frame_skip": args.frame_skip,
        "gray_scale": args.gray_scale,
        "reset_skip_sec": args.reset_skip_sec,
        "channels_last": True,
    }

    config_cls = IMPALAConfig if args.algo == "impala" else APPOConfig
    config = (
        config_cls()
        .environment(env=env_name, env_config=env_config, disable_env_checking=True)
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
        # One shared policy for all 4 in-window agents.
        .multi_agent(
            policies={SHARED_POLICY_ID},
            policy_mapping_fn=policy_mapping_fn,
        )
    )

    if args.lf2_window_resource:
        # Pin rollout EnvRunners to Windows nodes that run LF2.
        config = config.env_runners(
            num_cpus_per_env_runner=1,
            custom_resources_per_env_runner={"LF2_WINDOW": 1},
        )

    return config, env_name


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LF2 distributed async RL (APPO/IMPALA).")
    p.add_argument("--algo", choices=["appo", "impala"], default="appo")
    p.add_argument("--address", default=None, help="Ray cluster address, e.g. 'auto'.")
    p.add_argument("--windows-name", default="Little Fighter 2")
    p.add_argument("--player-ids", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--mode", choices=["mix", "picture", "info"], default="mix")
    p.add_argument("--frame-stack", type=int, default=4)
    p.add_argument("--frame-skip", type=int, default=1)
    p.add_argument("--gray-scale", action="store_true", default=True)
    p.add_argument("--no-gray-scale", dest="gray_scale", action="store_false")
    p.add_argument("--reset-skip-sec", type=int, default=2)

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
    ray.init(address=args.address)

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
