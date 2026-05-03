#!/usr/bin/env bash
# scripts/download_cross_domain_subsets.sh
# Download ~200-image subsets for cross-domain evaluation.
# Auto-downloads: VOC2012, Open Images v7 (via FiftyOne)
# Manual steps printed for: BDD100K, VisDrone, DOTA
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

NUM_IMAGES="${NUM_IMAGES:-200}"
SEED="${SEED:-42}"
PYTHON_BIN="${PYTHON_BIN:-python}"

log()  { echo "[cross-domain] $*"; }
count_images() { find "$1" -maxdepth 2 -type f 2>/dev/null | grep -cE '\.(jpg|jpeg|png)$' || echo 0; }

# ── helpers ──────────────────────────────────────────────────────────────────
sample_copy() {
  local src="$1" dst="$2" limit="$3"
  mkdir -p "$dst"
  SRC_DIR="$src" DST_DIR="$dst" LIMIT="$limit" SEED="$SEED" "$PYTHON_BIN" - <<'PY'
from pathlib import Path
import os, random, shutil
src = Path(os.environ['SRC_DIR'])
dst = Path(os.environ['DST_DIR'])
limit = int(os.environ['LIMIT'])
seed = int(os.environ['SEED'])
files = [p for p in src.rglob('*') if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png'}]
random.Random(seed).shuffle(files)
for i, p in enumerate(files[:limit]):
    tgt = dst / p.name
    if tgt.exists():
        tgt = dst / f"{i:04d}_{p.name}"
    shutil.copy2(p, tgt)
print(f"  copied {min(len(files), limit)} / {len(files)} files -> {dst}")
PY
}

mkdir -p \
  data/voc/VOCdevkit \
  data/bdd100k/images/100k/val \
  data/visdrone/VisDrone2019-DET-val/images \
  data/dota/val/images \
  data/openimages/validation \
  tmp_downloads

# ── 1. Pascal VOC 2012 ────────────────────────────────────────────────────────
log "=== VOC 2012 ==="
if [ ! -d data/voc/VOCdevkit/VOC2012/JPEGImages ]; then
  mkdir -p tmp_downloads/voc
  VOC_TAR="tmp_downloads/voc/VOCtrainval_11-May-2012.tar"
  if [ ! -f "$VOC_TAR" ]; then
    log "Downloading VOC2012 trainval (~2 GB)..."
    wget -c http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar -O "$VOC_TAR"
  fi
  log "Extracting VOC2012..."
  tar -xf "$VOC_TAR" -C data/voc
else
  log "VOC2012 already extracted, skipping."
fi
log "VOC2012 image count: $(count_images data/voc/VOCdevkit/VOC2012/JPEGImages)"

# ── 2. Open Images v7 via FiftyOne ───────────────────────────────────────────
log "=== Open Images v7 (validation subset, max=$NUM_IMAGES) ==="
if [ "$(count_images data/openimages/validation)" -lt "$NUM_IMAGES" ]; then
  log "Installing fiftyone if needed..."
  "$PYTHON_BIN" -m pip install -q fiftyone

  NUM_IMAGES="$NUM_IMAGES" "$PYTHON_BIN" - <<'PY'
import os, fiftyone.zoo as foz
max_s = int(os.environ['NUM_IMAGES'])
foz.load_zoo_dataset(
    'open-images-v7',
    split='validation',
    max_samples=max_s,
    dataset_dir='data/openimages/_fo_cache',
)
print(f"FiftyOne download done, max_samples={max_s}")
PY
  sample_copy data/openimages/_fo_cache data/openimages/validation "$NUM_IMAGES"
else
  log "Open Images already has $(count_images data/openimages/validation) images, skipping."
fi

# ── 3. BDD100K (manual via Kaggle CLI) ───────────────────────────────────────
log "=== BDD100K ==="
if [ "$(count_images data/bdd100k/images/100k/val)" -lt "$NUM_IMAGES" ]; then
  cat <<'MSG'

  [BDD100K] Please download manually via Kaggle CLI:
    pip install kaggle
    # Place kaggle.json at ~/.kaggle/kaggle.json and chmod 600
    kaggle datasets download -d marquis03/bdd100k -p tmp_downloads/bdd100k
    unzip -q tmp_downloads/bdd100k/bdd100k.zip -d tmp_downloads/bdd100k/unzipped

  Then run this one-liner to pick 200 val images:
    python - <<'PY'
from pathlib import Path; import random, shutil
src = Path('tmp_downloads/bdd100k/unzipped')
files = [p for p in src.rglob('*') if p.suffix.lower() in {'.jpg','.jpeg','.png'} and 'val' in str(p).lower()]
random.Random(42).shuffle(files)
dst = Path('data/bdd100k/images/100k/val'); dst.mkdir(parents=True, exist_ok=True)
[shutil.copy2(p, dst / (f'{i:04d}_' + p.name)) for i, p in enumerate(files[:200])]
print('BDD100K done:', len(list(dst.iterdir())))
PY

MSG
else
  log "BDD100K already has $(count_images data/bdd100k/images/100k/val) images, skipping."
fi

# ── 4. VisDrone ───────────────────────────────────────────────────────────────
log "=== VisDrone2019-DET-val ==="
if [ "$(count_images data/visdrone/VisDrone2019-DET-val/images)" -lt "$NUM_IMAGES" ]; then
  cat <<'MSG'

  [VisDrone] Download the val zip from the official release:
    https://github.com/VisDrone/VisDrone-Dataset
    (file: VisDrone2019-DET-val.zip)

  Place it under tmp_downloads/visdrone/, then run:
    unzip -q tmp_downloads/visdrone/VisDrone2019-DET-val.zip -d tmp_downloads/visdrone/unzipped
    python - <<'PY'
from pathlib import Path; import random, shutil
src = Path('tmp_downloads/visdrone/unzipped')
files = [p for p in src.rglob('*') if p.suffix.lower() in {'.jpg','.jpeg','.png'}]
random.Random(42).shuffle(files)
dst = Path('data/visdrone/VisDrone2019-DET-val/images'); dst.mkdir(parents=True, exist_ok=True)
[shutil.copy2(p, dst / (f'{i:04d}_' + p.name)) for i, p in enumerate(files[:200])]
print('VisDrone done:', len(list(dst.iterdir())))
PY

MSG
else
  log "VisDrone already has $(count_images data/visdrone/VisDrone2019-DET-val/images) images, skipping."
fi

# ── 5. DOTA ───────────────────────────────────────────────────────────────────
log "=== DOTA val ==="
if [ "$(count_images data/dota/val/images)" -lt "$NUM_IMAGES" ]; then
  cat <<'MSG'

  [DOTA] Download val images from: https://captain-whu.github.io/DOTA/dataset.html
  Extract to tmp_downloads/dota/, then run:
    python - <<'PY'
from pathlib import Path; import random, shutil
src = Path('tmp_downloads/dota')
files = [p for p in src.rglob('*') if p.suffix.lower() in {'.jpg','.jpeg','.png'}]
random.Random(42).shuffle(files)
dst = Path('data/dota/val/images'); dst.mkdir(parents=True, exist_ok=True)
[shutil.copy2(p, dst / (f'{i:04d}_' + p.name)) for i, p in enumerate(files[:200])]
print('DOTA done:', len(list(dst.iterdir())))
PY

MSG
else
  log "DOTA already has $(count_images data/dota/val/images) images, skipping."
fi

# ── Final summary ─────────────────────────────────────────────────────────────
echo ""
echo "=== Final dataset image counts ==="
for d in \
  "data/coco/val2017" \
  "data/voc/VOCdevkit/VOC2012/JPEGImages" \
  "data/bdd100k/images/100k/val" \
  "data/visdrone/VisDrone2019-DET-val/images" \
  "data/dota/val/images" \
  "data/openimages/validation"
do
  printf "  %-50s %s images\n" "$d" "$(count_images "$d")"
done
echo ""
echo "[done] Cross-domain data check complete."