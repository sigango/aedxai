#!/usr/bin/env bash
# Run cluster/task.sh tasks one after another on this machine, resumably.
#
#   cluster/run_chain.sh TASK[:SHARDS] [TASK[:SHARDS] ...]
#   e.g. cluster/run_chain.sh evidence-test-yolox:16 vlm-test-yolox:4 analyze-test
#
# Each task runs its shards in order; the runners skip finished images, so a rerun
# resumes. A failed task is retried up to RETRIES times (default 3) after 60 s, then
# the chain stops. Task output goes to logs/<task>.log; memory use is logged every
# 5 min to logs/<CHAIN_NAME>_mem.log. Keep the shard count of a task fixed once it
# has started. Start long chains with cluster/launch_unit.sh, not in tmux or a
# Jupyter terminal (both die with the session or the Jupyter service).
set -o pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi
chain="${CHAIN_NAME:-chain}"
retries="${RETRIES:-3}"
mkdir -p logs

log() { echo "== $(date '+%F %T') $*"; }

memlog() {  # every 5 min: system memory, swap, pipeline RSS, GPU memory of all compute processes
  while true; do
    echo "$(date '+%F %T')" \
      "$(free -g | awk '/^Mem:/ {printf "used=%sG avail=%sG", $3, $7} /^Swap:/ {printf " swap=%sG", $3}')" \
      "pipeline_rss=$(ps -eo rss=,args= | awk '/scripts\/run_(evidence|vlm_verification)\.py/ && !/awk/ {s += $1} END {printf "%.1fG", s / 1048576}')" \
      "gpu_all=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | awk '{s += $1} END {printf "%.1fG", s / 1024}')" \
      >> "logs/${chain}_mem.log"
    sleep 300
  done
}

run_task() {  # run_task TASK NUM_SHARDS
  local shard
  for ((shard = 0; shard < $2; shard++)); do
    bash cluster/task.sh "$1" "${shard}" "$2" || return 1
  done
}

if (( $# == 0 )); then
  echo "usage: cluster/run_chain.sh TASK[:SHARDS] ..." >&2
  exit 2
fi
if [[ -n "$(git status --porcelain)" ]]; then
  log "ABORT: uncommitted changes (run manifests would be marked dirty)"
  git status --short
  exit 1
fi
log "Start chain ${chain}: $*"
log "Commit $(git log --oneline -1); python $(command -v python)"
memlog &
memlog_pid=$!
trap 'kill "${memlog_pid}" 2>/dev/null' EXIT

for item in "$@"; do
  task="${item%%:*}"
  shards=1
  [[ "${item}" == *:* ]] && shards="${item##*:}"
  done_ok=0
  for ((attempt = 1; attempt <= retries; attempt++)); do
    log "${task} (${shards} shards) attempt ${attempt}"
    if run_task "${task}" "${shards}" 2>&1 | tee -a "logs/${task}.log"; then
      done_ok=1
      log "${task} done"
      break
    fi
    log "${task} failed (attempt ${attempt})"
    (( attempt < retries )) && sleep 60
  done
  if (( ! done_ok )); then
    log "Chain stopped at ${task}"
    exit 1
  fi
done
log "Chain finished"
