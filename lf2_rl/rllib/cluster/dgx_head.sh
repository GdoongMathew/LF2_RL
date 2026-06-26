#!/usr/bin/env bash
# Start the Ray head node + LF2 learner on the DGX10 (Linux, GPU).
#
# The DGX runs ONLY the learner (it cannot run the Windows-only game). Windows
# machines join this head as workers (see windows_worker.ps1) and run the LF2
# EnvRunners that advertise the custom `LF2_WINDOW` resource.
#
# Usage:
#   ./dgx_head.sh [RAY_PORT]
set -euo pipefail

RAY_PORT="${1:-6379}"
DASHBOARD_HOST="0.0.0.0"

# The head holds the GPU(s) for the learner. No LF2_WINDOW resource here so no
# rollout env is ever scheduled on the DGX.
ray start --head \
    --port "${RAY_PORT}" \
    --dashboard-host "${DASHBOARD_HOST}" \
    --num-gpus "${NUM_GPUS:-1}"

echo
echo "Ray head started on port ${RAY_PORT}."
echo "Head address for workers:  <DGX_IP>:${RAY_PORT}"
echo
echo "Next: on each Windows LF2 machine run:"
echo "    .\\windows_worker.ps1 -HeadAddress <DGX_IP>:${RAY_PORT}"
echo
echo "Then launch training on the head:"
echo "    python -m lf2_rl.rllib.train --address auto \\"
echo "        --num-env-runners <num_windows> --lf2-window-resource \\"
echo "        --num-learners 1 --num-gpus-per-learner 1"
