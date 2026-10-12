#!/usr/bin/env bash
# Start cluster/run_chain.sh as a transient systemd user unit, so a long chain
# survives logouts and tmux/Jupyter restarts, and an OOM kill of one task does not
# stop the unit (the chain retries the task). Needs lingering for the user
# (`loginctl show-user "$USER" -p Linger` -> yes).
#
#   cluster/launch_unit.sh UNIT TASK[:SHARDS] [TASK[:SHARDS] ...]
#   e.g. cluster/launch_unit.sh aedxai-test evidence-test-yolox:16 evidence-test-frcnn:16 \
#          vlm-test-yolox:4 vlm-test-frcnn:4 analyze-test
#
# Follow it with:
#   systemctl --user status UNIT;  tail -f logs/UNIT.log;  tail logs/UNIT_mem.log
# Stop it with: systemctl --user stop UNIT
set -euo pipefail
unit="${1:?usage: cluster/launch_unit.sh UNIT TASK[:SHARDS] ...}"
shift
(( $# > 0 )) || { echo "usage: cluster/launch_unit.sh UNIT TASK[:SHARDS] ..." >&2; exit 2; }
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
mkdir -p logs

if [[ -n "$(git status --porcelain)" ]]; then
  echo "ABORT: uncommitted changes (run manifests would be marked dirty)" >&2
  git status --short >&2
  exit 1
fi
if pgrep -f "scripts/run_(evidence|vlm_verification)\.py" >/dev/null; then
  echo "ABORT: a pipeline process is already running:" >&2
  pgrep -af "scripts/run_(evidence|vlm_verification)\.py" >&2
  exit 1
fi

systemd-run --user --unit "${unit}" --collect \
  -p OOMPolicy=continue \
  -p WorkingDirectory="${PWD}" \
  -p StandardOutput=append:"${PWD}/logs/${unit}.log" \
  -p StandardError=append:"${PWD}/logs/${unit}.log" \
  --setenv=CHAIN_NAME="${unit}" \
  bash "${PWD}/cluster/run_chain.sh" "$@"
echo "Started user unit ${unit}. Log: logs/${unit}.log, memory: logs/${unit}_mem.log"
