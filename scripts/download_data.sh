#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATA_DIR="${PROJECT_ROOT}/data"
COCO_DIR="${DATA_DIR}/coco"
VAL_DIR="${COCO_DIR}/val2017"
ANN_DIR="${COCO_DIR}/annotations"
CHECKPOINT_DIR="${DATA_DIR}/checkpoints"
TMP_DIR="$(mktemp -d)"

VAL_ZIP_URL="http://images.cocodataset.org/zips/val2017.zip"
ANN_ZIP_URL="http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
YOLOX_WEIGHTS_URL="https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth"

cleanup() {
  rm -rf "${TMP_DIR}"
}

download_file() {
  local url="$1"
  local output_path="$2"

  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --output "${output_path}" "${url}"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "${output_path}" "${url}"
  else
    echo "Error: curl or wget is required to download assets." >&2
    exit 1
  fi
}

print_checksum() {
  local file_path="$1"

  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "${file_path}"
  else
    shasum -a 256 "${file_path}"
  fi
}

trap cleanup EXIT

mkdir -p "${VAL_DIR}" "${ANN_DIR}" "${CHECKPOINT_DIR}"

VAL_ZIP_PATH="${TMP_DIR}/val2017.zip"
ANN_ZIP_PATH="${TMP_DIR}/annotations_trainval2017.zip"
YOLOX_WEIGHTS_PATH="${CHECKPOINT_DIR}/yolox_s.pth"

echo "Downloading COCO val2017 images..."
download_file "${VAL_ZIP_URL}" "${VAL_ZIP_PATH}"

echo "Downloading COCO annotations..."
download_file "${ANN_ZIP_URL}" "${ANN_ZIP_PATH}"

echo "Downloading YOLOX-S weights..."
download_file "${YOLOX_WEIGHTS_URL}" "${YOLOX_WEIGHTS_PATH}"

if [ -z "$(ls -A "${VAL_DIR}" 2>/dev/null)" ]; then
  echo "Extracting COCO val2017 images..."
  unzip -q "${VAL_ZIP_PATH}" -d "${COCO_DIR}"
else
  echo "Skipping image extraction because ${VAL_DIR} is already populated."
fi

if [ ! -f "${ANN_DIR}/instances_val2017.json" ]; then
  echo "Extracting COCO annotations..."
  unzip -q "${ANN_ZIP_PATH}" -d "${COCO_DIR}"
else
  echo "Skipping annotation extraction because ${ANN_DIR} already contains instances_val2017.json."
fi

echo "Checksums:"
print_checksum "${VAL_ZIP_PATH}"
print_checksum "${ANN_ZIP_PATH}"
print_checksum "${YOLOX_WEIGHTS_PATH}"

echo "Download complete."

