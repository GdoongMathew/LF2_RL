#!/usr/bin/env bash
# Pull a full Algorithm checkpoint from the DGX learner via SSH/scp.
#
# Use this when:
#   * Training is already stopped (--windows-checkpoint-dir only syncs while
#     workers are still connected to the head).
#   * You want a full ``algo.save()`` checkpoint (algorithm state + config +
#     module) rather than just the RLModule weights that train.py's Ray
#     broadcast writes.
#   * You want to retrieve a specific past iteration that was overwritten on
#     the Windows side.
#
# Run from the Windows machine in Git Bash / WSL / native OpenSSH client.
#
# Usage (env vars):
#   DGX_HOST=dgx10.example.com \
#   DGX_USER=mathew \
#   DGX_PATH=/data/lf2/LF2_RL_Model \
#   LOCAL_PATH=C:/LF2_RL_Checkpoint \
#       ./fetch_checkpoint.sh
#
# All four variables are required so nothing leaks a wrong default into a
# different deployment. Set DGX_SSH_KEY if you need a non-default key.

set -euo pipefail

DGX_HOST="${DGX_HOST:-}"
DGX_USER="${DGX_USER:-}"
DGX_PATH="${DGX_PATH:-}"
LOCAL_PATH="${LOCAL_PATH:-}"
DGX_SSH_KEY="${DGX_SSH_KEY:-}"

usage() {
    cat <<EOF
fetch_checkpoint.sh — scp a full RLlib algorithm checkpoint from the DGX
                      learner to a Windows-local directory.

Required environment variables:
  DGX_HOST     DGX hostname or IP (e.g. dgx10.example.com)
  DGX_USER     SSH username on the DGX
  DGX_PATH     Source directory on the DGX (matches --checkpoint-dir
               passed to train.py; default \"LF2_RL_Model\" lives in the
               driver's cwd)
  LOCAL_PATH   Destination directory on this Windows machine

Optional:
  DGX_SSH_KEY  Path to a non-default private key

After download, point ``lf2_rl.rllib.eval --checkpoint-path`` at the
``rl_module`` sub-folder of the downloaded checkpoint (see eval.py
help), or use ``Algorithm.from_checkpoint(LOCAL_PATH)`` to fully
restore the algorithm.
EOF
}

for var in DGX_HOST DGX_USER DGX_PATH LOCAL_PATH; do
    if [[ -z "${!var}" ]]; then
        echo "ERROR: \$${var} is required." >&2
        usage
        exit 1
    fi
done

SCP_OPTS=(-r)
if [[ -n "${DGX_SSH_KEY}" ]]; then
    SCP_OPTS+=(-i "${DGX_SSH_KEY}")
fi

echo "Pulling LF2 checkpoint:"
echo "  src   : ${DGX_USER}@${DGX_HOST}:${DGX_PATH}"
echo "  dest  : ${LOCAL_PATH}"
mkdir -p "${LOCAL_PATH}"

scp "${SCP_OPTS[@]}" "${DGX_USER}@${DGX_HOST}:${DGX_PATH}" "${LOCAL_PATH}/"

echo "Done. To play with this checkpoint:"
echo "  python -m lf2_rl.rllib.eval \\"
echo "      --checkpoint-path '${LOCAL_PATH}/$(basename "${DGX_PATH}")/learner_group/learner/rl_module' \\"
echo "      --mode mix --player-ids 0 1 2 3"
