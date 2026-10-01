#!/usr/bin/env bash
# Run one task (optionally one shard of it). Every cluster job goes through here,
# so the command for each experiment is defined exactly once.
#
#   cluster/task.sh <task> [shard_index] [num_shards]
#
# Tasks (DET = yolox | frcnn):
#   setup-manifests                 COCO + BDD100K manifests (CPU)
#   setup-bdd-annotations           BDD100K -> COCO-80 annotations (CPU)
#   evidence-smoke-DET              3 dev images, full config (notebook 06 smoke test)
#   vlm-smoke-DET                   VLM pass over the smoke run
#   evidence-pilot-DET              dev split, full evidence config
#   evidence-calib-DET              calib split, lite config
#   evidence-test-DET               test split, lite config
#   evidence-bdd-DET                BDD100K test split, lite config
#   vlm-pilot-DET | vlm-test-DET | vlm-bdd-DET   VLM pass over an evidence run
#   selector-DET                    retrain the legacy selector from the existing progress CSV
#   baseline-DET-METHOD             legacy fixed-method baseline on the COCO test split
#   aedxai-DET-v1                   legacy selector pipeline, feedback off, VLM off
#   aedxai-DET-v1orig               legacy selector pipeline with original settings
#   aedxai-DET-cascade              always-G-CAME -> LIME cascade (needs calibrated tau)
#   analyze-pilot | analyze-test | analyze-bdd | calibrate-tau   CPU analyses
set -euo pipefail

TASK="${1:?usage: cluster/task.sh <task> [shard_index] [num_shards]}"
SHARD="${2:-0}"
NSHARDS="${3:-1}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
ENV_FILE="${AEDXAI_ENV:-cluster/cluster.env}"
if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing ${ENV_FILE}; copy cluster/cluster.env.example and edit it." >&2
  exit 2
fi
# shellcheck disable=SC1090
source "${ENV_FILE}"
export PYTHONUNBUFFERED=1

run() {
  echo "[task ${TASK} shard ${SHARD}/${NSHARDS}] $*"
  # shellcheck disable=SC2086
  ${RUN_PREFIX} "${PYTHON}" "$@"
}

det_name() {
  case "$1" in
    yolox) echo "yolox-s" ;;
    frcnn) echo "fasterrcnn_resnet50_fpn_v2" ;;
    *) echo "unknown detector key '$1'" >&2; exit 2 ;;
  esac
}

evidence() {  # evidence <split> <det_key> <config> <images> <ann> <manifest> <run_prefix> [extra args...]
  local split="$1" det_key="$2" det; det="$(det_name "$2")"
  local config="$3" images="$4" ann="$5" manifest="$6" prefix="$7"
  shift 7
  # Optional per-detector image cap, e.g. NUM_IMAGES_test_frcnn=1000 (0/empty = whole split).
  # BDD100K also uses the "test" split, so it has its own cap: NUM_IMAGES_bdd_<det>.
  local cap_var="NUM_IMAGES_${split}_${det_key}"
  if [[ "${prefix}" == "bdd" ]]; then cap_var="NUM_IMAGES_bdd_${det_key}"; fi
  if [[ "${prefix}" == "smoke" ]]; then SMOKE_IMAGES="${SMOKE_IMAGES:-3}"; cap_var="SMOKE_IMAGES"; fi
  run scripts/run_evidence.py \
    --images-dir "${images}" --annotations "${ann}" --image-list "${manifest}" --split "${split}" \
    --detector-model "${det}" --evidence-config "${config}" \
    --run-name "${prefix}_${split}_${det}" --num-shards "${NSHARDS}" --shard-index "${SHARD}" \
    --num-images "${!cap_var:-0}" \
    ${MAX_EXPLAINED:+--max-explained "${MAX_EXPLAINED}"} "$@"
}

vlm() {  # vlm <run_name> <images>
  run scripts/run_vlm_verification.py --run-dir "results/evidence/$1" --images-dir "$2" \
    --num-shards "${NSHARDS}" --shard-index "${SHARD}"
}

case "${TASK}" in
  setup-manifests)
    run scripts/make_manifests.py --images-dir "${COCO_IMAGES}" --annotations "${COCO_ANN}" \
      --splits dev:300,calib:1000,test:2000 --output "${MANIFEST}"
    if [[ -f "${BDD_ANN}" ]]; then
      run scripts/make_manifests.py --images-dir "${BDD_IMAGES}" --annotations "${BDD_ANN}" \
        --exclude-oracle-csvs "" --splits calib:300,test:1000 --output "${BDD_MANIFEST}"
    fi
    ;;
  setup-bdd-annotations)
    run scripts/convert_annotations.py --format bdd100k --source "${BDD_LABELS}" \
      --images-dir "${BDD_IMAGES}" --output "${BDD_ANN}"
    ;;
  # Smoke test: a few dev images with the full config (notebook 06).
  evidence-smoke-*) evidence dev "${TASK##*-}" config/evidence_config.yaml "${COCO_IMAGES}" "${COCO_ANN}" "${MANIFEST}" smoke ;;
  vlm-smoke-*) vlm "smoke_dev_$(det_name "${TASK##*-}")" "${COCO_IMAGES}" ;;
  evidence-pilot-*) evidence dev "${TASK##*-}" config/evidence_config.yaml "${COCO_IMAGES}" "${COCO_ANN}" "${MANIFEST}" coco ;;
  # The calib split only feeds tau calibration, so it needs G-CAME composites alone.
  evidence-calib-*) evidence calib "${TASK##*-}" config/evidence_config_lite.yaml "${COCO_IMAGES}" "${COCO_ANN}" "${MANIFEST}" coco \
                      --methods gcame --controls "" --no-maps ;;
  evidence-test-*) evidence test "${TASK##*-}" config/evidence_config_lite.yaml "${COCO_IMAGES}" "${COCO_ANN}" "${MANIFEST}" coco ;;
  evidence-bdd-*) evidence test "${TASK##*-}" config/evidence_config_lite.yaml "${BDD_IMAGES}" "${BDD_ANN}" "${BDD_MANIFEST}" bdd ;;
  vlm-pilot-*) vlm "coco_dev_$(det_name "${TASK##*-}")" "${COCO_IMAGES}" ;;
  vlm-test-*) vlm "coco_test_$(det_name "${TASK##*-}")" "${COCO_IMAGES}" ;;
  vlm-bdd-*) vlm "bdd_test_$(det_name "${TASK##*-}")" "${BDD_IMAGES}" ;;
  selector-*)
    # Relabel the existing 2,000-row progress CSV with the evaluator-scale oracle and
    # retrain (no new oracle images are processed once --target-detections is reached).
    det="$(det_name "${TASK##*-}")"
    run scripts/train_selector.py --detector-model "${det}" --max-images 1000 \
      --target-detections 2000 --oracle-mode --resume
    ;;
  baseline-*)
    rest="${TASK#baseline-}"; det="$(det_name "${rest%%-*}")"; method="${rest#*-}"
    run scripts/run_baseline.py --images-dir "${COCO_IMAGES}" --image-list "${MANIFEST}" --split test \
      --num-images "${LEGACY_NUM_IMAGES:-1000}" \
      --detector-model "${det}" --xai-method "${method}" --output "results/v2_legacy/baseline"
    ;;
  aedxai-*)
    rest="${TASK#aedxai-}"; det="$(det_name "${rest%%-*}")"; variant="${rest#*-}"
    common=(scripts/run_experiments.py --images-dir "${COCO_IMAGES}" --image-list "${MANIFEST}" --split test
            --num-images "${LEGACY_NUM_IMAGES:-1000}"
            --detector-model "${det}" --output "results/v2_legacy/aedxai_${variant}_${det}")
    case "${variant}" in
      v1) run "${common[@]}" --max-iterations 1 --no-vlm --escalation off ;;
      v1orig) run "${common[@]}" --escalation off ;;
      cascade) run "${common[@]}" --max-iterations 1 --no-vlm --fixed-method gcame --escalation on ;;
      *) echo "unknown aedxai variant ${variant}" >&2; exit 2 ;;
    esac
    ;;
  calibrate-tau)
    run scripts/calibrate_tau.py --run-dirs results/evidence/coco_calib_yolox-s \
      results/evidence/coco_calib_fasterrcnn_resnet50_fpn_v2 --percentile 30 \
      --output results/analysis/tau_calibration.json
    ;;
  analyze-pilot|analyze-test|analyze-bdd)
    case "${TASK}" in
      analyze-pilot) prefix=coco_dev; cfg=config/evidence_config.yaml ;;
      analyze-test) prefix=coco_test; cfg=config/evidence_config_lite.yaml ;;
      analyze-bdd) prefix=bdd_test; cfg=config/evidence_config_lite.yaml ;;
    esac
    runs=(results/evidence/${prefix}_yolox-s results/evidence/${prefix}_fasterrcnn_resnet50_fpn_v2)
    existing=(); for r in "${runs[@]}"; do [[ -d "$r" ]] && existing+=("$r"); done
    if (( ${#existing[@]} == 0 )); then echo "No evidence runs found for ${prefix}" >&2; exit 1; fi
    run scripts/analyze_evidence.py --run-dirs "${existing[@]}" --evidence-config "${cfg}" \
      --output-dir "results/analysis/evidence_${prefix}"
    run scripts/analyze_metric_validity.py --run-dirs "${existing[@]}" \
      --output-dir "results/analysis/metric_validity_${prefix}"
    ;;
  *)
    echo "Unknown task '${TASK}'" >&2
    exit 2
    ;;
esac
