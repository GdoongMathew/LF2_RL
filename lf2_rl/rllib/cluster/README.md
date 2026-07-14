# LF2 distributed async RL — cluster setup

A heterogeneous Ray cluster: the **DGX10 (Linux, GPU)** is the head node and runs
the **learner**; one or more **Windows machines** run Little Fighter 2 and host the
rollout **EnvRunners** (the "actors"). Experience flows from the Windows actors to
the DGX learner; updated policy weights are broadcast back — an asynchronous
APPO/IMPALA actor–learner loop.

```
   Windows #1 (LF2)            Windows #2 (LF2)            DGX10 (GPU)
   EnvRunner (4 agents)        EnvRunner (4 agents)        Ray head + Learner
   LF2_WINDOW:1, GPU:1   ─exp─┐  LF2_WINDOW:1, GPU:1 ─exp─┐ num_gpus:1
        ▲   (GPU inference)    └────────┐    ▲             └──────►  train()
        └───────weights──────────────────┴────┴──────────weights broadcast
```

Inference (policy forward pass) runs on each Windows worker — optionally on a
GPU there. **No gradient** is ever computed on Windows; the only place
gradients run is the DGX learner.

## 0. One-time prerequisite — Ray Windows multi-node opt-in

Ray's multi-node cluster mode with a Windows worker is **experimental**
and gated behind an env var. You must export
`RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=1` on **both** the DGX head and any
Windows worker, otherwise `ray start --address=<head>` on Windows bails
out with *"Multi-node Ray clusters are not supported on Windows and
OSX"*.

`dgx_head.sh` and `windows_worker.ps1` set this variable for the spawned
`ray start` process automatically. When invoking `train.py` on the DGX
directly (e.g. without going through the shell wrapper), set it too:

```bash
RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=1 python -m lf2_rl.rllib.train ...
```

## 1. Install (on every node)

```
pip install -e .[rl]              # DGX learner side (Linux): cross-platform deps only
pip install -e .[rl,windows]      # Windows rollout side: + game I/O (pywin32/pymem/mss/...)
```

`platform_system == 'Windows'` markers on the `windows` extra make it a
no-op if installed on Linux, so `[rl,windows]` is safe everywhere.

## 2. Start the head on the DGX

```bash
NUM_GPUS=1 ./lf2_rl/rllib/cluster/dgx_head.sh 6379
```

Note the printed head address `<DGX_IP>:6379`.

## 3. Join each Windows LF2 machine

Start LF2 (VS Mode, players selected), keep it focused, then:

```powershell
# With a Windows-side GPU for policy inference:
.\lf2_rl\rllib\cluster\windows_worker.ps1 -HeadAddress <DGX_IP>:6379 -NumGpus 1

# CPU-only inference (no Windows GPU):
.\lf2_rl\rllib\cluster\windows_worker.ps1 -HeadAddress <DGX_IP>:6379
```

Each Windows node advertises `LF2_WINDOW: 1` (and optionally its GPU).

## 4. Launch training (on the DGX head)

```bash
python -m lf2_rl.rllib.train \
    --address auto \
    --algo appo \
    --num-env-runners <number_of_windows_machines> \
    --num-learners 1 --num-gpus-per-learner 1 \
    --num-gpus-per-env-runner 1 \
    --mode mix --player-ids 0 1 2 3
```

What the defaults already enforce (no flag needed):

* **`--lf2-window-resource`** is on by default — every rollout EnvRunner is
  pinned to a node that advertises the `LF2_WINDOW` resource, so the
  learner stays on the DGX GPU and rollouts stay on Windows. Disable with
  `--no-lf2-window-resource` only for a single-machine smoke test on a
  Windows dev box.
* **`--num-env-runners` ≥ 1** is enforced — `0` would push the rollout
  onto a local worker inside the DGX driver and immediately crash on
  `import lf2_gym.windows.*`.

`--num-gpus-per-env-runner` is the inference-GPU share. Set to `0` to keep
inference on CPU; otherwise it must be ≤ the GPU count each Windows worker
advertised in step 3.

## Behaviour on failure

* If LF2 isn't running / not focused on a Windows node, the
  `Lf2GameController` ctor on that worker raises after
  `DEFAULT_FIRST_FRAME_TIMEOUT = 30s` (no more silent infinite wait); Ray
  will mark the worker dead and you'll see the error in the head logs
  instead of an indefinite hang.

## Scaling

* **In-window parallelism:** each EnvRunner already yields 4 agents' experience
  per window via the shared policy (`--player-ids 0 1 2 3`).
* **Across machines:** add more Windows workers and raise `--num-env-runners`.

## Checkpointing

`--checkpoint-dir` (default `LF2_RL_Model`) on the DGX is where
`algo.save()` writes a **full algorithm checkpoint** every
`--checkpoint-freq` iterations and on exit. It's local to the DGX driver
process — useful for resuming training on the same DGX.

For getting weights back to Windows there are two complementary paths,
covering the in-training and offline cases.

### Path A — In-training Ray broadcast (no SSH needed)

Pass `--windows-checkpoint-dir <windows-local-path>` to `train.py`:

```bash
python -m lf2_rl.rllib.train \
    --address auto \
    --num-env-runners <num_windows> \
    --num-learners 1 --num-gpus-per-learner 1 \
    --num-gpus-per-env-runner 1 \
    --checkpoint-dir LF2_RL_Model \
    --windows-checkpoint-dir C:/lf2_models
```

After every `algo.save()` (and on shutdown), the DGX driver fans out a
Ray actor RPC that has each Windows EnvRunner write its currently-loaded
`RLModule` to `C:/lf2_models/latest` on **its own** local filesystem.
The transfer uses Ray's existing TCP channel — no SSH, no scp, no
shared storage. Each Windows machine ends up with its own copy of the
latest weights, ready for standalone playback.

### Path B — Offline scp pull (when training has stopped)

Use the SSH-only environment to grab a full algorithm checkpoint
on-demand from the Windows side:

```bash
# Git Bash / WSL / OpenSSH on the Windows machine
DGX_HOST=dgx10.example.com \
DGX_USER=mathew \
DGX_PATH=/data/lf2/LF2_RL_Model \
LOCAL_PATH=C:/LF2_RL_Checkpoint \
    ./fetch_checkpoint.sh
```

Required when:

* Training is already stopped (workers are no longer connected).
* You want a full `algo.save()` checkpoint (full algorithm state for
  resume) rather than just the RLModule weights from path A.
* You want a specific past iteration that's already been overwritten on
  the Windows side.

## Standalone playback on Windows

Once Windows has weights (via path A or path B), use `lf2_rl.rllib.eval`
to drive an `Lf2ParallelEnv` step-by-step with the loaded policy. No
Ray, no DGX involvement:

```powershell
# Path A: load directly from the windows-checkpoint-dir
python -m lf2_rl.rllib.eval `
    --checkpoint-path C:/lf2_models/latest `
    --mode mix --player-ids 0 1 2 3

# Path B: drill into the algorithm checkpoint's module sub-tree
python -m lf2_rl.rllib.eval `
    --checkpoint-path C:/LF2_RL_Checkpoint/LF2_RL_Model/learner_group/learner/rl_module `
    --mode mix --player-ids 0 1 2 3
```

`eval.py` uses `forward_inference` only — no gradient is ever computed
on Windows.
