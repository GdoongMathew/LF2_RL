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
    if args.num_players:
        env_config["num_players"] = args.num_players

    # Declare obs/action spaces purely from config so RLlib never has to
    # construct an env on the driver to probe them.
    obs_space, act_space = build_spaces_from_config(env_config, channels_last=True)
    # Print so any driver-vs-worker shape mismatch is immediately visible.
    # (RLlib rejects mismatched obs at rollout time with a "Observation ...
    # outside given space" error — best to catch it before build_algo.)
    print(f"[spec] declared obs_space = {obs_space}")
    print(f"[spec] declared act_space = {act_space}")

    config_cls = IMPALAConfig if args.algo == "impala" else APPOConfig
    config = (
        config_cls()
        # RLlib 2.55's *new* API stack default Catalog can only auto-build
        # encoders for pure 1-D Box (MLP) or 3-D Box uint8 (CNN); it raises
        # "No default encoder config for obs space=Dict(...)" for our
        # mix-mode observation. It also has a downstream connector-v2 bug
        # that trips on Discrete action spaces in multi-agent setups.
        # Switch to the *old* API stack whose ``ComplexInputNet`` handles
        # Dict observations natively and doesn't use connector_v2. Old
        # stack is deprecated in RLlib 2.x but still functional; migrating
        # to a custom RLModule on the new stack is a follow-up.
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment(
            env=LF2_PARALLEL_ENV,
            env_config=env_config,
            observation_space=obs_space,
            action_space=act_space,
            disable_env_checking=True,
            clip_rewards=args.clip_rewards,
        )
        .framework("torch")
        # Async rollout workers (the LF2 "actors"). Place them on Windows nodes
        # that advertise the custom `LF2_WINDOW` resource (see cluster scripts).
        .env_runners(
            num_env_runners=args.num_env_runners,
            num_envs_per_env_runner=1,
            rollout_fragment_length=args.rollout_fragment_length,
            # Windows EnvRunner can use a GPU slice for policy inference (no
            # gradient is computed on it — that stays on the DGX learner).
            num_gpus_per_env_runner=args.num_gpus_per_env_runner,
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
            entropy_coeff=args.entropy_coeff,
            grad_clip=args.grad_clip,
            vf_loss_coeff=args.vf_loss_coeff,
            # Old API's VisionNet only ships default conv_filters for
            # Atari-standard shapes (42/64/84/10). Our LF2 obs is
            # ~199x400x1 after downscale — spell out a Nature-style
            # stack extended with one more stride-2 conv to shrink the
            # feature map before flattening. ComplexInputNet applies
            # these to each image leaf of the Dict obs; the info leaf
            # goes through fcnet_hiddens.
            model={
                "conv_filters": [
                    [32, [8, 8], 4],
                    [64, [4, 4], 2],
                    [64, [3, 3], 2],
                    [64, [3, 3], 1],
                ],
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
            },
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
        # Pin rollout EnvRunners to Windows nodes that run LF2. Without this
        # pin Ray's scheduler is free to land them on the DGX head, where
        # ``import lf2_gym.windows.*`` immediately fails — so this is on by
        # default and only opt-out via ``--no-lf2-window-resource``.
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
        "the standard 800x600 LF2 window + --downscale. Pass the actual "
        "post-downscale height when your LF2 window is not 800x600.",
    )
    p.add_argument("--img-w", type=int, default=None, help="Override observation image width.")
    p.add_argument(
        "--num-players",
        type=int,
        default=None,
        help="Total active player slots in the LF2 scene (agents + CPU bots). "
        "The Info observation carries one row per active slot. Defaults to "
        "len(--player-ids); pass the real count when CPU bots share the "
        "scene (otherwise the driver's declared Info shape won't match the "
        "env's actual output and RLlib rejects the obs).",
    )

    p.add_argument("--num-env-runners", type=int, default=1)
    p.add_argument("--num-learners", type=int, default=1)
    p.add_argument("--num-gpus-per-learner", type=float, default=1.0)
    p.add_argument(
        "--num-gpus-per-env-runner",
        type=float,
        default=0.0,
        help="GPU share each Windows EnvRunner gets for policy inference "
        "(default 0 = CPU inference). Set to e.g. 0.25 to share one GPU "
        "across 4 workers on a multi-GPU Windows node.",
    )
    # On-by-default: keep rollout EnvRunners pinned to Windows nodes that
    # advertise the ``LF2_WINDOW`` resource. Opt out only when running a
    # single-machine smoke test on a Windows dev box.
    p.add_argument(
        "--lf2-window-resource",
        dest="lf2_window_resource",
        action="store_true",
        default=True,
        help="Pin rollout EnvRunners to nodes with the LF2_WINDOW resource "
        "(default: on).",
    )
    p.add_argument(
        "--no-lf2-window-resource",
        dest="lf2_window_resource",
        action="store_false",
        help="Disable the LF2_WINDOW pin (single-machine dev only).",
    )

    p.add_argument("--rollout-fragment-length", type=int, default=50)
    p.add_argument("--train-batch-size", type=int, default=500)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument(
        "--entropy-coeff",
        type=float,
        default=0.01,
        help="Entropy regularization weight for the policy loss. Raise to "
        "0.05-0.1 if you see ``entropy`` collapse to ~0 in training logs "
        "(policy becomes deterministic and stops exploring).",
    )
    p.add_argument(
        "--grad-clip",
        type=float,
        default=40.0,
        help="Global L2 gradient-norm clip. If training logs show "
        "``grad_gnorm`` pinned at exactly this value every iteration, the "
        "clip is dominating — raise it (e.g. 100, 200) or set to a very "
        "large number to effectively disable.",
    )
    p.add_argument(
        "--vf-loss-coeff",
        type=float,
        default=0.5,
        help="Multiplier applied to the value-function loss before adding "
        "to total_loss. Default 0.5 assumes reward-scale ~O(1). With big "
        "raw rewards (episode returns in the hundreds) ``vf_loss`` can "
        "dwarf ``policy_loss`` and dominate the gradient — drop this to "
        "0.05 or 0.01 to rebalance and let the policy actually learn.",
    )
    p.add_argument(
        "--clip-rewards",
        action="store_true",
        default=False,
        help="Clip per-step rewards to [-1, 1] before advantage computation "
        "(RLlib built-in ``clip_rewards=True``). Sledgehammer fix when raw "
        "rewards are huge and destabilizing the critic — advantages become "
        "bounded and value-function targets stay in a learnable range.",
    )

    p.add_argument("--stop-timesteps", type=int, default=6_000_000)
    p.add_argument("--checkpoint-dir", default="LF2_RL_Model")
    p.add_argument("--checkpoint-freq", type=int, default=50)
    p.add_argument(
        "--windows-checkpoint-dir",
        default=None,
        help="If set, after every algo.save() each Windows EnvRunner also "
        "writes its local RLModule weights to <path>/latest on the worker's "
        "*local* filesystem (Windows-side). No SSH / scp needed — uses Ray's "
        "existing actor RPC. Consume from ``lf2_rl.rllib.eval`` on the same "
        "Windows machine for standalone playback. Use a Windows-style path "
        "with forward slashes, e.g. ``C:/lf2_models``.",
    )
    return p.parse_args(argv)


def _save_module_to_local(env_runner, base_path: str, iteration: int) -> str | None:
    """Persist the EnvRunner's current policy weights to a Windows-local path.

    Runs on each Windows worker process via ``foreach_env_runner``. Uses
    only the in-memory weights the learner has already broadcast to this
    worker, so it doesn't touch the network filesystem.

    Handles both API stacks:

    * **New API stack** — ``env_runner.module`` is a ``RLModule`` /
      ``MultiRLModule``; call ``save_to_path`` to write a full checkpoint
      directory that :func:`ray.rllib.core.rl_module.RLModule.from_checkpoint`
      can reload.
    * **Old API stack** — ``env_runner.policy_map`` maps policy id to
      ``Policy``; pickle each policy's ``get_weights()`` into
      ``<final>/<policy_id>.pkl``. Load with a manual
      ``policy.set_weights(pickle.load(...))``.
    """
    import pathlib
    import pickle

    final = pathlib.Path(base_path) / "latest"
    final.mkdir(parents=True, exist_ok=True)

    if hasattr(env_runner, "module") and env_runner.module is not None:
        env_runner.module.save_to_path(str(final))
        return f"iter_{iteration:06d} (new API) -> {final}"

    if hasattr(env_runner, "policy_map"):
        for pid, policy in env_runner.policy_map.items():
            with open(final / f"{pid}.pkl", "wb") as f:
                pickle.dump(policy.get_weights(), f)
        return f"iter_{iteration:06d} (old API, pickled weights) -> {final}"

    return None


def _broadcast_module_to_windows(algo, base_path: str, iteration: int) -> None:
    """Have each Windows EnvRunner snapshot its current RLModule locally.

    New API stack only — the ``env_runner.module.save_to_path`` route
    doesn't exist on the old-stack ``RolloutWorker``, which stores state
    on ``policy_map[SHARED_POLICY_ID]`` instead. On old-stack builds this
    call is a no-op so the training loop doesn't crash.
    """
    runner_group = getattr(algo, "env_runner_group", None)
    if runner_group is None or not hasattr(runner_group, "foreach_env_runner"):
        print(
            "  [windows-checkpoint] skipped — old API stack build has no "
            "env_runner_group / module accessor. Use algo.save() from the "
            "driver + fetch_checkpoint.sh from Windows for now."
        )
        return
    try:
        paths = runner_group.foreach_env_runner(
            func=lambda w: _save_module_to_local(w, base_path, iteration),
            local_env_runner=False,
        )
    except AttributeError as exc:
        # Same reason as above — .module attribute absent on old stack.
        print(f"  [windows-checkpoint] skipped: {exc}")
        return
    for p in paths:
        if p:
            print(f"  windows-checkpoint: {p}")


def main(argv=None) -> None:
    import ray

    args = parse_args(argv)

    # ``num_env_runners == 0`` would push the rollout onto a local worker
    # in the driver process (the DGX). That worker would then top-level
    # import ``lf2_gym.windows.*`` and crash, since the driver is Linux.
    # Force the user to opt explicitly into the remote-worker model.
    if args.num_env_runners < 1:
        raise SystemExit(
            f"--num-env-runners must be >= 1 (got {args.num_env_runners}); "
            "the LF2 env is Windows-only and cannot run on the DGX driver."
        )

    ray.init(address=args.address)

    # Register the LF2 env creator on the driver. Ray Tune's registry is
    # backed by a shared cluster-wide actor, so this single call
    # propagates to every worker automatically — no per-worker setup
    # hook needed (which used to break on the old-stack ``RolloutWorker``
    # spawn path).
    #
    # ``worker_register`` defers the Windows-specific
    # ``lf2_gym.windows.*`` import into the creator body, so importing
    # it here is safe on the DGX Linux driver too.
    import lf2_rl.rllib.worker_register  # noqa: F401 (side-effect import)

    # Preflight: surface a missing LF2_WINDOW resource *before* RLlib goes
    # to the autoscaler with a cryptic "No available node types..." error.
    # The advertise happens on the Windows side via ``windows_worker.ps1``
    # which passes ``--resources '{"LF2_WINDOW": N}'`` to ``ray start``.
    if args.lf2_window_resource:
        lf2_capacity = sum(
            node.get("Resources", {}).get("LF2_WINDOW", 0)
            for node in ray.nodes()
            if node.get("Alive", False)
        )
        if lf2_capacity < args.num_env_runners:
            raise SystemExit(
                f"Cluster has {lf2_capacity:g} LF2_WINDOW resources but "
                f"--num-env-runners={args.num_env_runners} requested. "
                "Either start more Windows workers via "
                "lf2_rl/rllib/cluster/windows_worker.ps1, or pass "
                "--no-lf2-window-resource for a single-machine dev run on "
                "this Windows box."
            )

    config, _ = build_config(args)
    # ``build()`` was renamed to ``build_algo()`` in recent RLlib versions;
    # both still work but ``build()`` emits a DeprecationWarning.
    algo = config.build_algo() if hasattr(config, "build_algo") else config.build()

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
                if args.windows_checkpoint_dir:
                    _broadcast_module_to_windows(
                        algo, args.windows_checkpoint_dir, iteration
                    )

            if ts >= args.stop_timesteps:
                break
    finally:
        algo.save(args.checkpoint_dir)
        if args.windows_checkpoint_dir:
            _broadcast_module_to_windows(
                algo, args.windows_checkpoint_dir, iteration
            )
        algo.stop()
        ray.shutdown()


if __name__ == "__main__":
    main()
