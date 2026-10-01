#!/usr/bin/env bash
# Launch experiment stages on SLURM (array jobs + afterok dependencies) or locally.
#
#   cluster/submit.sh <stage> [<stage> ...]
#
# Stages, in order (see RESULTS_AND_PLAN.md Part B for what each one decides):
#   setup     manifests (+ BDD100K annotation conversion if labels exist)      CPU
#   smoke     3 dev images per detector, full config (see notebook 06)         GPU
#   vlm-smoke VLM pass over the smoke runs                                      GPU
#   pilot     evidence on COCO dev (300 imgs, full config), both detectors      GPU
#   vlm-pilot VLM verification on the pilot runs                                GPU
#   calib     evidence on COCO calib split (lite)                               GPU
#   test      evidence on COCO test split (lite)                                GPU
#   vlm-test  VLM verification on the test runs                                 GPU
#   bdd       evidence on BDD100K test split (lite)                             GPU
#   vlm-bdd   VLM verification on BDD100K                                       GPU
#   selector  retrain the legacy selector from existing progress CSVs          GPU (short)
#   legacy    v1 baselines (4 methods x 2 detectors) + v1 pipeline variants     GPU
#   cascade   always-G-CAME -> LIME cascade with calibrated tau (after calib)   GPU
#   analyze-pilot | analyze-test | analyze-bdd | calibrate-tau                  CPU
#
# BACKEND=slurm (default from cluster.env): consecutive stages in one call are
# chained with --dependency=afterok. BACKEND=local: stages run one after another,
# shards run in parallel over LOCAL_GPUS x PROCS_PER_GPU slots.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
ENV_FILE="${AEDXAI_ENV:-cluster/cluster.env}"
# shellcheck disable=SC1090
source "${ENV_FILE}"
mkdir -p logs
PREVIOUS_JOBS=""

launch() {  # launch <task> <num_shards> <kind: gpu|cpu>
  local task="$1" shards="$2" kind="$3"
  if [[ "${BACKEND}" == "slurm" ]]; then
    local args=(--parsable --job-name "aedxai-${task}" --output "logs/%x_%A_%a.out"
                --time "${SLURM_TIME}" --cpus-per-task "${SLURM_CPUS}" --mem "${SLURM_MEM}"
                --array "0-$((shards - 1))")
    [[ -n "${SLURM_PARTITION}" ]] && args+=(--partition "${SLURM_PARTITION}")
    [[ -n "${SLURM_ACCOUNT}" ]] && args+=(--account "${SLURM_ACCOUNT}")
    [[ "${kind}" == "gpu" ]] && args+=(--gres "${SLURM_GRES}")
    [[ -n "${PREVIOUS_JOBS}" ]] && args+=(--dependency "afterok:${PREVIOUS_JOBS}")
    local id
    id="$(sbatch "${args[@]}" --export "ALL,AEDXAI_ENV=${ENV_FILE}" \
          --wrap "cluster/task.sh ${task} \${SLURM_ARRAY_TASK_ID} ${shards}")"
    echo "submitted ${task} (${shards} shards) -> job ${id}"
    STAGE_JOBS="${STAGE_JOBS:+${STAGE_JOBS}:}${id}"
  else
    IFS=',' read -r -a gpus <<< "${LOCAL_GPUS}"
    local slots=$(( ${#gpus[@]} * PROCS_PER_GPU )) pids=() shard
    for ((shard = 0; shard < shards; shard++)); do
      local gpu="${gpus[$(( shard % ${#gpus[@]} ))]}"
      CUDA_VISIBLE_DEVICES="${gpu}" cluster/task.sh "${task}" "${shard}" "${shards}" \
        > "logs/${task}_shard${shard}.log" 2>&1 &
      pids+=($!)
      if (( ${#pids[@]} >= slots )); then wait "${pids[0]}"; pids=("${pids[@]:1}"); fi
    done
    for pid in ${pids[@]+"${pids[@]}"}; do wait "${pid}"; done
    echo "finished ${task} (${shards} shards); logs in logs/${task}_shard*.log"
  fi
}

stage() {
  STAGE_JOBS=""
  case "$1" in
    setup)
      if [[ -f "${BDD_LABELS}" ]]; then launch setup-bdd-annotations 1 cpu; PREVIOUS_JOBS="${STAGE_JOBS}"; STAGE_JOBS=""; fi
      launch setup-manifests 1 cpu ;;
    smoke) for d in yolox frcnn; do launch "evidence-smoke-${d}" 1 gpu; done ;;
    vlm-smoke) for d in yolox frcnn; do launch "vlm-smoke-${d}" 1 gpu; done ;;
    pilot) for d in yolox frcnn; do launch "evidence-pilot-${d}" "${SHARDS_PILOT}" gpu; done ;;
    vlm-pilot) for d in yolox frcnn; do launch "vlm-pilot-${d}" "${SHARDS_VLM}" gpu; done ;;
    calib) for d in yolox frcnn; do launch "evidence-calib-${d}" "${SHARDS_CALIB}" gpu; done ;;
    test) for d in yolox frcnn; do launch "evidence-test-${d}" "${SHARDS_TEST}" gpu; done ;;
    vlm-test) for d in yolox frcnn; do launch "vlm-test-${d}" "${SHARDS_VLM}" gpu; done ;;
    bdd) for d in yolox frcnn; do launch "evidence-bdd-${d}" "${SHARDS_XDOMAIN}" gpu; done ;;
    vlm-bdd) for d in yolox frcnn; do launch "vlm-bdd-${d}" "${SHARDS_VLM}" gpu; done ;;
    selector) for d in yolox frcnn; do launch "selector-${d}" 1 gpu; done ;;
    legacy)
      # Per-detector method and variant lists (cluster.env); defaults = every v1 row.
      for d in yolox frcnn; do
        methods_var="LEGACY_METHODS_${d}" variants_var="LEGACY_VARIANTS_${d}"
        for m in ${!methods_var-gradcam gcame dclose lime}; do launch "baseline-${d}-${m}" 1 gpu; done
        for v in ${!variants_var-v1 v1orig}; do launch "aedxai-${d}-${v}" 1 gpu; done
      done ;;
    cascade) for d in ${CASCADE_DETECTORS-yolox frcnn}; do launch "aedxai-${d}-cascade" 1 gpu; done ;;
    analyze-pilot|analyze-test|analyze-bdd|calibrate-tau) launch "$1" 1 cpu ;;
    *) echo "unknown stage '$1'" >&2; exit 2 ;;
  esac
  PREVIOUS_JOBS="${STAGE_JOBS}"
}

if [[ $# -eq 0 ]]; then
  sed -n '2,20p' "$0"
  exit 1
fi
for name in "$@"; do stage "${name}"; done
