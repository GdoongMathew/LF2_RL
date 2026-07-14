#!/usr/bin/env bash
# Start the Ray head node + LF2 learner on the DGX10 (Linux, GPU).
#
# The DGX runs ONLY the learner (it cannot run the Windows-only game). Windows
# machines join this head as workers (see windows_worker.ps1) and run the LF2
# EnvRunners that advertise the custom `LF2_WINDOW` resource.
#
# NOTE: Ray's multi-node mode with a Windows worker is experimental and gated
#       behind `RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=1`. The variable must be set
#       on BOTH the head (this script) and each worker (windows_worker.ps1
#       does it automatically).
#
# NOTE: If workers reach this node over a private overlay network like
#       Tailscale / WireGuard / ZeroTier, the OS default route may pick a
#       *different* NIC than the one workers can actually use. Pass the
#       reachable IP via NODE_IP_ADDRESS so Ray binds and advertises there:
#
#           NODE_IP_ADDRESS=100.79.61.7 ./dgx_head.sh 6379
#
#       Without this, ``ray start --address=<head>`` on the worker will
#       ``Test-NetConnection`` fine but ``ConnectionError`` at Ray-client
#       handshake time because the head published an unroutable endpoint.
#
# Usage:
#   [NODE_IP_ADDRESS=<ip>] [NUM_GPUS=<n>] ./dgx_head.sh [RAY_PORT]
set -euo pipefail

RAY_PORT="${1:-6379}"
DASHBOARD_HOST="0.0.0.0"

# Opt-in to Ray's experimental Windows/macOS multi-node mode. Without this
# the Windows worker's ``ray start --address=...`` immediately bails out
# with "Multi-node Ray clusters are not supported on Windows and OSX".
export RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=1

node_ip_opts=()
if [[ -n "${NODE_IP_ADDRESS:-}" ]]; then
    node_ip_opts+=(--node-ip-address "${NODE_IP_ADDRESS}")
    echo "Binding Ray head to ${NODE_IP_ADDRESS} (from NODE_IP_ADDRESS)."
else
    echo "NODE_IP_ADDRESS is unset — Ray will auto-detect the interface,"
    echo "which often picks the primary LAN NIC instead of a private overlay"
    echo "(Tailscale/WireGuard). Set NODE_IP_ADDRESS to the head IP that"
    echo "Windows workers can reach if you see ConnectionError at join time."
fi

# The head holds the GPU(s) for the learner. No LF2_WINDOW resource here so no
# rollout env is ever scheduled on the DGX.
ray start --head \
    --port "${RAY_PORT}" \
    --dashboard-host "${DASHBOARD_HOST}" \
    --num-gpus "${NUM_GPUS:-1}" \
    "${node_ip_opts[@]}"

echo
echo "Ray head started on port ${RAY_PORT}."
echo "Head address for workers:  ${NODE_IP_ADDRESS:-<DGX_IP>}:${RAY_PORT}"
echo
echo "Next: on each Windows LF2 machine run:"
echo "    .\\windows_worker.ps1 -HeadAddress ${NODE_IP_ADDRESS:-<DGX_IP>}:${RAY_PORT} \\"
echo "        -NodeIpAddress <THIS_WINDOWS_TAILSCALE_IP> -NumGpus 1"
echo
echo "Then launch training on the head (LF2_WINDOW pin is on by default):"
echo "    RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=1 python -m lf2_rl.rllib.train \\"
echo "        --address ${NODE_IP_ADDRESS:-<DGX_IP>}:${RAY_PORT} \\"
echo "        --num-env-runners <num_windows> \\"
echo "        --num-learners 1 --num-gpus-per-learner 1 \\"
echo "        --num-gpus-per-env-runner 1"
