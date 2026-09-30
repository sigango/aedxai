#!/usr/bin/env bash
# Verify the cluster environment before submitting anything (run on a GPU node).
#   cluster/env_check.sh            full check incl. fast tests
#   cluster/env_check.sh --no-tests
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
# shellcheck disable=SC1090
source "${AEDXAI_ENV:-cluster/cluster.env}"
status=0

echo "== nvidia-smi"; nvidia-smi || status=1

echo "== python / CUDA stack"
${RUN_PREFIX} "${PYTHON}" - <<'PY' || status=1
import platform, torch
print("machine:", platform.machine(), "| torch:", torch.__version__, "| CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("GPU:", p.name, "| VRAM GB:", round(p.total_memory / 1e9, 1), "| devices:", torch.cuda.device_count())
PY

echo "== imports"
${RUN_PREFIX} "${PYTHON}" - <<'PY' || status=1
import importlib, sys
failed = []
for name in ["torch", "torchvision", "transformers", "qwen_vl_utils", "yolox", "pycocotools", "cv2",
             "captum", "skimage", "sklearn", "scipy", "pandas", "yaml"]:
    try:
        importlib.import_module(name); print(f"{name}: OK")
    except Exception as exc:  # noqa: BLE001
        failed.append(name); print(f"{name}: FAIL -> {type(exc).__name__}: {exc}")
try:
    import bitsandbytes  # noqa: F401
    print("bitsandbytes: OK (only needed for int4/int8 VLM)")
except Exception as exc:  # noqa: BLE001
    print(f"bitsandbytes: not usable ({exc}); keep vlm_config quantization: fp16")
sys.exit(1 if failed else 0)
PY

echo "== data"
for path in "${COCO_IMAGES}" "${COCO_ANN}" data/checkpoints/yolox_s.pth; do
  if [[ -e "${path}" ]]; then echo "OK      ${path}"; else echo "MISSING ${path}"; status=1; fi
done
[[ -e "${BDD_IMAGES}" ]] && echo "OK      ${BDD_IMAGES}" || echo "absent  ${BDD_IMAGES} (only needed for the bdd stage)"

if [[ "${1:-}" != "--no-tests" ]]; then
  echo "== compile + fast tests"
  ${RUN_PREFIX} "${PYTHON}" -m compileall -q src scripts tests || status=1
  ${RUN_PREFIX} "${PYTHON}" -m pytest -q -m "not slow" tests || status=1
  echo "== model-dependent tests (GPU)"
  ${RUN_PREFIX} "${PYTHON}" -m pytest -q tests/test_detector.py tests/test_xai_methods.py || status=1
fi

exit "${status}"
