"""Standalone LF2 playback using a trained RLModule (Windows-only).

Loads the per-worker RLModule that ``train.py --windows-checkpoint-dir``
wrote (via Ray actor RPC, no SSH) or a module sub-tree pulled from the
DGX with ``cluster/fetch_checkpoint.sh``, and drives an
``Lf2ParallelEnv`` step-by-step on this Windows machine. Does **not**
connect to Ray; runs entirely locally — useful for qualitative
evaluation / demos once training has produced something interesting.

The "training" never happens here: only ``forward_inference`` to pick
actions for the live LF2 window.

Usage::

    python -m lf2_rl.rllib.eval \\
        --checkpoint-path C:/lf2_models/latest \\
        --mode mix --player-ids 0 1 2 3
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from lf2_gym.windows.parallel_env import Lf2ParallelEnv
from lf2_rl.rllib.spec import SHARED_POLICY_ID, ChannelsLastWrapper


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--checkpoint-path",
        required=True,
        type=Path,
        help="Path to a saved RLModule (the directory ``module.save_to_path`` "
        "wrote). For a full algorithm checkpoint pulled via "
        "fetch_checkpoint.sh, point at ``<ckpt>/learner_group/learner/rl_module``.",
    )
    p.add_argument(
        "--policy-id",
        default=SHARED_POLICY_ID,
        help=f"Sub-module key inside a MultiRLModule (default: {SHARED_POLICY_ID}).",
    )
    p.add_argument("--windows-name", default="Little Fighter 2")
    p.add_argument("--player-ids", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--mode", choices=["mix", "picture", "info"], default="mix")
    p.add_argument("--frame-stack", type=int, default=4)
    p.add_argument("--frame-skip", type=int, default=1)
    p.add_argument("--downscale", type=int, default=2)
    p.add_argument("--gray-scale", action="store_true", default=True)
    p.add_argument("--no-gray-scale", dest="gray_scale", action="store_false")
    p.add_argument("--reset-skip-sec", type=int, default=2)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Greedy argmax over policy logits (default). "
        "Use --no-deterministic to sample from the distribution.",
    )
    p.add_argument("--no-deterministic", dest="deterministic", action="store_false")
    return p.parse_args(argv)


def _load_shared_module(checkpoint_path: Path, policy_id: str):
    """Load the saved RLModule and drill into the shared-policy sub-module."""
    from ray.rllib.core.rl_module.rl_module import RLModule

    if not checkpoint_path.exists():
        raise SystemExit(f"--checkpoint-path does not exist: {checkpoint_path}")

    loaded = RLModule.from_checkpoint(str(checkpoint_path))
    # MultiRLModule supports ``__getitem__`` for sub-modules. A single-policy
    # checkpoint is returned directly.
    if hasattr(loaded, "__getitem__"):
        try:
            return loaded[policy_id]
        except KeyError:
            pass  # not a multi-module — fall through
    return loaded


def _to_batch(obs: Any):
    """Wrap one agent's obs into the (B=1) tensor batch RLModule expects."""
    import torch

    if isinstance(obs, dict):
        return {k: torch.as_tensor(np.asarray(v)).unsqueeze(0) for k, v in obs.items()}
    return torch.as_tensor(np.asarray(obs)).unsqueeze(0)


def _compute_action(module, obs: Any, deterministic: bool) -> int:
    """Extract a discrete action from RLModule.forward_inference output."""
    import torch

    batch = {"obs": _to_batch(obs)}
    with torch.no_grad():
        out = module.forward_inference(batch)

    # New API stack returns ``action_dist_inputs`` (logits for Discrete).
    # Older paths may return ``actions`` directly.
    if "actions" in out:
        return int(out["actions"][0].item())
    logits = out.get("action_dist_inputs")
    if logits is None:
        raise RuntimeError(
            "RLModule.forward_inference returned unknown keys: " f"{list(out.keys())}"
        )
    if deterministic:
        return int(torch.argmax(logits[0]).item())
    probs = torch.softmax(logits[0], dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def main(argv=None) -> None:
    args = parse_args(argv)
    print(f"Loading RLModule from {args.checkpoint_path} (policy={args.policy_id}) ...")
    module = _load_shared_module(args.checkpoint_path, args.policy_id)
    print("  module ready.")

    env: Any = Lf2ParallelEnv(
        windows_name=args.windows_name,
        player_ids=tuple(args.player_ids),
        downscale=args.downscale,
        frame_stack=args.frame_stack,
        frame_skip=args.frame_skip,
        reset_skip_sec=args.reset_skip_sec,
        gray_scale=args.gray_scale,
        mode=args.mode,
    )
    # train.py wraps the env with ChannelsLastWrapper (channels_last=True),
    # so eval must do the same to feed obs in the layout the conv net was
    # trained on.
    env = ChannelsLastWrapper(env)

    try:
        for ep in range(args.episodes):
            obs, _ = env.reset()
            steps = 0
            ep_rewards: dict[str, float] = {a: 0.0 for a in obs}
            while True:
                actions = {
                    agent: _compute_action(module, ob, args.deterministic)
                    for agent, ob in obs.items()
                }
                obs, rewards, terms, truncs, infos = env.step(actions)
                for a, r in rewards.items():
                    ep_rewards[a] = ep_rewards.get(a, 0.0) + r
                steps += 1
                if (terms and all(terms.values())) or (truncs and all(truncs.values())):
                    break
            print(f"[ep {ep + 1}/{args.episodes}] steps={steps} rewards={ep_rewards}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
