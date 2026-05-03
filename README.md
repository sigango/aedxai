# AED-XAI: Adaptive Explanation-Driven Object Detection

AED-XAI is a research pipeline for **closed-loop, auto-explainable object detection**. It detects objects, asks a vision-language model to judge detection context, adaptively selects the most suitable XAI method per detection, evaluates explanation quality without human explanation annotations, and feeds low explanation-quality signals back into detector post-processing thresholds.

The core idea is simple but powerful: explanation quality depends on context. A small high-confidence object in a clean scene may be best served by G-CAME, while a low-confidence object in clutter may need a slower but more faithful perturbation method such as D-CLOSE. AED-XAI learns this selection offline from oracle labels and uses it online from pre-explanation features only.

## Pipeline

```text
RGB image
   |
   v
+------------------+
| DetectorWrapper  |  YOLOX-S or Faster R-CNN
+------------------+
   |
   v
+------------------+
| VLMJudge         |  Qwen2.5-VL detection quality and scene context
+------------------+
   |
   v
+------------------+
| XAISelector      |  MLP or rule fallback selects GradCAM/G-CAME/D-CLOSE/LIME
+------------------+
   |
   v
+------------------+
| XAI Explainer    |  saliency map for each detection
+------------------+
   |
   v
+------------------+
| AutoEvaluator    |  PG, EBPG, OA, Sparsity, Composite
+------------------+
   |
   v
+------------------+
| FeedbackLoop     |  adaptive thresholding and detector refinement
+------------------+
```

## Current Implementation

- `DetectorWrapper` supports YOLOX-S and torchvision Faster R-CNN.
- `VLMJudge` uses Qwen2.5-VL with robust JSON parsing, retry prompts, and VLM unload support.
- XAI methods include GradCAM, G-CAME, D-CLOSE, and LIME adapted for object detection.
- `XAISelector` supports a trained 7-feature MLP and a rule-based fallback.
- Selector oracle labels are generated offline by running all candidate XAI methods and choosing the best composite score.
- `AutoEvaluator` computes annotation-free metrics using detector boxes as pseudo-ground-truth.
- `AdaptiveThreshold` supports fixed, percentile, and learned threshold modes.
- `FeedbackLoop` adjusts confidence and NMS thresholds based on explanation quality.
- `AEDXAIPipeline` orchestrates the complete closed loop and keeps the VLM resident across batch calls.
- `pipeline_io.py` serializes per-image JSON summaries, detection overlays, and saliency PNG/NPY outputs.
- Scripts support full experiments, fixed-method baselines, selector training, detector comparison, selector-size ablations, and figure generation.

## Repository Layout

```text
config/
  detector_config.yaml       # detector settings and thresholds
  vlm_config.yaml            # Qwen2.5-VL model, quantization, prompts
  xai_config.yaml            # GradCAM, G-CAME, D-CLOSE, LIME settings
  eval_config.yaml           # metrics, composite weights, feedback thresholds

src/
  detector.py                # Detection, DetectorWrapper, scene complexity
  vlm_judge.py               # Assessment, VLMJudge
  xai_selector.py            # XAISelectorMLP, XAISelector
  evaluator.py               # EvalResult, AutoEvaluator
  threshold.py               # AdaptiveThreshold
  feedback_loop.py           # FeedbackLoop, FeedbackResult
  pipeline.py                # AEDXAIPipeline, PipelineResult
  pipeline_io.py             # result serialization helpers
  utils.py                   # image I/O, logging, COCO classes, bbox utilities
  xai_methods/
    base.py
    gradcam.py
    gcame.py
    dclose.py
    lime_det.py

scripts/
  download_data.sh
  train_selector.py
  run_experiments.py
  run_baseline.py
  compare_detectors.py
  run_cross_domain.py
  ablation_selector_size.py
  generate_figures.py

notebooks/
  00_setup_and_data.ipynb
  01_explore_detections.ipynb
  02_vlm_judge_analysis.ipynb
  03_xai_comparison.ipynb
  04_run_all_experiments.ipynb
  05_results_visualization.ipynb

tests/
  test_*.py
```

## Hardware Targets

Recommended runtime:

```text
Linux
Python 3.10+
NVIDIA GPU
CUDA 12.x or newer driver-compatible runtime
24GB VRAM minimum for full RTX 3090/4090 workflows
```

Known project targets:

```text
NVIDIA RTX 3090 / 4090 class GPU, AMD64, 24GB VRAM
NVIDIA DGX Spark / GB10, ARM64, 128GB unified VRAM, CUDA 13.0
```

For DGX Spark or other ARM64 NVIDIA systems, prefer an NVIDIA PyTorch or NGC container. Avoid replacing the container-provided `torch` and `torchvision` wheels unless you intentionally know the replacement supports the platform.

## Installation

### Option A: NVIDIA PyTorch Container

```bash
docker run --gpus all -it --rm --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$PWD":/workspace/aedxai \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  -v "$HOME/.cache/torch":/root/.cache/torch \
  -w /workspace/aedxai \
  nvcr.io/nvidia/pytorch:25.11-py3 \
  bash
```

Inside the container:

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-build-isolation -e .
grep -vE '^(torch|torchvision)' requirements.txt > /tmp/requirements-no-torch.txt
python -m pip install -r /tmp/requirements-no-torch.txt
python -m pip install --no-build-isolation -r requirements-yolox.txt
```

### Option B: Virtualenv

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-build-isolation -e .
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation -r requirements-yolox.txt
```

YOLOX must be installed after PyTorch is available. Use `--no-build-isolation` so the YOLOX build can import the already-installed `torch`.

## Verify Environment

```bash
nvidia-smi
nvcc --version
```

```bash
python - <<'PY'
import platform, torch
print("machine:", platform.machine())
print("torch:", torch.__version__)
print("torch CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM GB:", torch.cuda.get_device_properties(0).total_memory / 1e9)
PY
```

```bash
python - <<'PY'
mods = ["torch", "torchvision", "transformers", "qwen_vl_utils", "pycocotools", "yolox"]
for name in mods:
    try:
        __import__(name)
        print(f"{name}: OK")
    except Exception as exc:
        print(f"{name}: FAIL -> {type(exc).__name__}: {exc}")
PY
```

## Data

Download COCO val2017, annotations, and YOLOX-S weights:

```bash
chmod +x scripts/download_data.sh
bash scripts/download_data.sh
```

Expected files:

```text
data/coco/val2017/
data/coco/annotations/instances_val2017.json
data/checkpoints/yolox_s.pth
```

## Quick Start

### Detector-Only Smoke Test

```bash
python - <<'PY'
from src.detector import DetectorWrapper
from src.utils import load_image

detector = DetectorWrapper("yolox-s", config_path="config/detector_config.yaml")
detector.load_model()
image = load_image("data/coco/val2017/000000000139.jpg")
detections = detector.detect(image)
print("detections:", len(detections))
print(detections[:3])
detector.unload_model()
PY
```

### Full Pipeline on One Image

```bash
python - <<'PY'
from src.pipeline import AEDXAIPipeline

pipeline = AEDXAIPipeline(
    detector_config_path="config/detector_config.yaml",
    vlm_config_path="config/vlm_config.yaml",
    xai_config_path="config/xai_config.yaml",
    eval_config_path="config/eval_config.yaml",
)
pipeline.setup()
result = pipeline.run_on_image("data/coco/val2017/000000000139.jpg")
print("composite:", result.composite_score)
print("metadata:", result.metadata)
print("selector reasoning keys:", list(result.selector_reasoning.keys())[:5])
pipeline.shutdown()
PY
```

## Experiments

### Train the XAI Selector

Small smoke run:

```bash
python scripts/train_selector.py \
  --detector-model yolox-s \
  --max-images 20 \
  --target-detections 100 \
  --checkpoint-every 25 \
  --oracle-mode
```

Train detector-specific selector checkpoints for both supported detectors:

```bash
for detector in yolox-s fasterrcnn_resnet50_fpn_v2; do
  python scripts/train_selector.py \
    --detector-model "$detector" \
    --max-images 1000 \
    --target-detections 2000 \
    --checkpoint-every 100 \
    --oracle-mode \
    --resume
done
```

By default, selector CSVs and checkpoints are detector-specific:

```text
results/xai_selector_training_data_yolox-s.csv
results/xai_selector_training_data_fasterrcnn_resnet50_fpn_v2.csv
data/checkpoints/xai_selector_yolox-s.pth
data/checkpoints/xai_selector_fasterrcnn_resnet50_fpn_v2.pth
```

### Run Full AED-XAI

```bash
python scripts/run_experiments.py \
  --images-dir data/coco/val2017 \
  --num-images 1000 \
  --output results/aedxai \
  --detector-model yolox-s \
  --seed 42
```

Outputs:

```text
results/aedxai/results.csv
results/aedxai/summary.json
```

### Run Fixed-Method Baselines

```bash
python scripts/run_baseline.py \
  --images-dir data/coco/val2017 \
  --num-images 1000 \
  --output results/baseline \
  --detector-model yolox-s \
  --xai-method gradcam \
  --seed 42
```

Output:

```text
results/baseline/baseline_gradcam_results.csv
```

Repeat with `--xai-method` set to:

```text
gradcam
gcame
dclose
lime
```

For Faster R-CNN baselines, use `--detector-model fasterrcnn_resnet50_fpn_v2`;
the output filename includes the detector name to avoid overwriting YOLOX rows.

### Compare Detectors

```bash
python scripts/compare_detectors.py \
  --images-dir data/coco/val2017 \
  --max-images 1000 \
  --detectors yolox-s,fasterrcnn_resnet50_fpn_v2 \
  --output-dir results
```

Outputs:

```text
results/compare_detectors_per_image.csv
results/compare_detectors_summary.csv
```

### Cross-Domain Evaluation

Run AED-XAI and fixed-method baselines on multiple image domains. The default
dataset list expects folders for COCO, VOC, BDD100K, VisDrone, DOTA, and
OpenImages; missing folders are skipped unless `--strict` is passed.

```bash
python scripts/run_cross_domain.py \
  --num-images 200 \
  --methods aedxai,gradcam,gcame,dclose,lime \
  --output-dir results \
  --recursive
```

Custom domains use comma-separated `NAME=IMAGE_DIR` entries:

```bash
python scripts/run_cross_domain.py \
  --datasets "COCO=data/coco/val2017,VOC=data/voc/VOCdevkit/VOC2012/JPEGImages,VisDrone=data/visdrone/VisDrone2019-DET-val/images" \
  --num-images 200 \
  --output-dir results
```

Outputs:

```text
results/cross_domain.csv
results/cross_domain_per_image.csv
results/cross_domain_summary.csv
results/cross_domain_summary.json
```

### Selector Training-Size Ablation

```bash
python scripts/ablation_selector_size.py \
  --training-csv results/xai_selector_training_data_yolox-s.csv \
  --sizes 50,100,200,500,1000 \
  --seeds 42,123,456
```

Notebook 04 runs this for both detector-specific selector CSVs and writes a
combined detector-aware ablation table.

Outputs:

```text
results/ablation_selector_size.csv
results/ablation_selector_size_summary.csv
results/figures/ablation_selector_size.png
```

### Generate Figures

```bash
python scripts/generate_figures.py \
  --results-root results \
  --figures-dir results/figures
```

If real experiment CSVs are missing, the figure script can synthesize placeholder data so figure layouts remain runnable while experiments finish.

## Metrics

AED-XAI evaluates explanations without human explanation annotations. Detector boxes act as pseudo-ground-truth for localization-style metrics.

Implemented metrics:

```text
PG       : Pointing Game, peak saliency inside bbox
EBPG     : Energy-Based Pointing Game, saliency mass inside bbox
OA       : insertion_auc - deletion_auc
Sparsity : Gini concentration of saliency
Composite: equal-weight combination of EBPG, normalized OA, and Sparsity
```

Current default composite weights:

```yaml
composite_weights:
  ebpg: 0.333333
  oa: 0.333333
  sparsity: 0.333333
```

`OA` is mapped into `[0, 1]` before weighting. The default feedback threshold mode is percentile-based:

```yaml
feedback:
  threshold_mode: "percentile"
  threshold_percentile: 40
```

## Result Serialization

Use `src.pipeline_io.save_pipeline_result` to save a structured per-image output directory:

```python
from src.pipeline_io import save_pipeline_result

save_pipeline_result(result, "results/per_image", save_saliency_npy=True)
```

Each image directory includes:

```text
result.json
detections.png
det_000_saliency.png
det_000_saliency.npy   # optional
```

## Notebooks

- `00_setup_and_data.ipynb`: setup, dependency install, COCO download, import checks.
- `01_explore_detections.ipynb`: qualitative YOLOX-S detection exploration.
- `02_vlm_judge_analysis.ipynb`: VLM quality, scene complexity, false positive, and latency analysis.
- `03_xai_comparison.ipynb`: side-by-side XAI comparison across four methods.
- `04_run_all_experiments.ipynb`: end-to-end experiment runner, detector-specific selector training, 1000-image evaluation, baseline comparison, detector comparison, cross-domain evaluation, and result export.
- `05_results_visualization.ipynb`: publication-style figures and ablation tables from saved CSV/JSON outputs.

Some notebooks cache expensive VLM/XAI results under `data/`.

## Testing

Fast checks:

```bash
python -m compileall src scripts tests
pytest tests/test_threshold.py tests/test_evaluator.py tests/test_feedback_loop.py tests/test_pipeline.py -q
```

Selector tests:

```bash
pytest tests/test_selector.py -q
```

Model-dependent tests:

```bash
pytest tests/test_detector.py -q
pytest tests/test_xai_methods.py -q
pytest tests/test_vlm_judge.py -m "not slow" -q
```

Slow GPU tests:

```bash
pytest -m slow
```

## Common Issues

### YOLOX build fails

Install YOLOX after PyTorch with build isolation disabled:

```bash
python -m pip install --no-build-isolation -r requirements-yolox.txt
```

### VLM INT4 fails on ARM64

Switch to FP16 in `config/vlm_config.yaml`:

```yaml
quantization: "fp16"
```

### CUDA is unavailable inside Docker

Check host and container GPU visibility:

```bash
nvidia-smi
docker run --rm --gpus=all nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04 nvidia-smi
```

### Notebook kernel imports wrong environment

Select the same Python environment where `torch`, `torchvision`, `pycocotools`, YOLOX, and AED-XAI are installed.

## Citation

```bibtex
@inproceedings{aedxai2026,
  title     = {AED-XAI: Adaptive Explanation-Driven Object Detection},
  author    = {TODO},
  booktitle = {TODO},
  year      = {2026}
}
```
