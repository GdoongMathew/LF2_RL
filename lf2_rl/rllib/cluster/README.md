# LF2 distributed async RL — cluster setup

A heterogeneous Ray cluster: the **DGX10 (Linux, GPU)** is the head node and runs
the **learner**; one or more **Windows machines** run Little Fighter 2 and host the
rollout **EnvRunners** (the "actors"). Experience flows from the Windows actors to
the DGX learner; updated policy weights are broadcast back — an asynchronous
APPO/IMPALA actor–learner loop.

```
   Windows #1 (LF2)            Windows #2 (LF2)            DGX10 (GPU)
   EnvRunner (4 agents)        EnvRunner (4 agents)        Ray head + Learner
   LF2_WINDOW:1  ───exp──┐     LF2_WINDOW:1 ───exp──┐      num_gpus:1
        ▲                 └──────────┐    ▲          └──────────►  train()
        └───────weights──────────────┴────┴──────────weights broadcast
```

## 1. Install (on every node)

```
pip install -e .[rl]      # ray[rllib], torch, pettingzoo, + game deps on Windows
```

The DGX does **not** need the Windows game deps; install at least
`pip install -e .[rl]` there for `ray` + `torch` + the training code.

## 2. Start the head on the DGX

```bash
NUM_GPUS=1 ./lf2_rl/rllib/cluster/dgx_head.sh 6379
```

Note the printed head address `<DGX_IP>:6379`.

## 3. Join each Windows LF2 machine

Start LF2 (VS Mode, players selected), keep it focused, then:

```powershell
.\lf2_rl\rllib\cluster\windows_worker.ps1 -HeadAddress <DGX_IP>:6379
```

Each Windows node advertises `LF2_WINDOW: 1`.

## 4. Launch training (on the DGX head)

```bash
python -m lf2_rl.rllib.train \
    --address auto \
    --algo appo \
    --num-env-runners <number_of_windows_machines> \
    --lf2-window-resource \
    --num-learners 1 --num-gpus-per-learner 1 \
    --mode mix --player-ids 0 1 2 3
```

`--lf2-window-resource` pins every rollout EnvRunner to a node with the
`LF2_WINDOW` resource, guaranteeing envs only run on Windows machines while the
learner stays on the DGX GPU.

## Scaling

* **In-window parallelism:** each EnvRunner already yields 4 agents' experience
  per window via the shared policy (`--player-ids 0 1 2 3`).
* **Across machines:** add more Windows workers and raise `--num-env-runners`.

## Checkpointing

Checkpoints are written to `--checkpoint-dir` every `--checkpoint-freq`
iterations and on exit; resume by loading the saved Algorithm checkpoint.
