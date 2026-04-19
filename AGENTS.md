# AED-XAI Agent Guide

This file is the working context map for AI coding agents and collaborators.
Read it before changing code. It captures the actual implementation state,
runtime assumptions, research contracts, and safe commands for the AED-XAI
project.

## Project Summary

AED-XAI means **Adaptive Explanation-Driven Object Detection with XAI**.

The project detects objects, judges detection quality with a vision-language
model, selects an explanation method per detection, evaluates explanation
quality without human explanation annotations, and feeds low explanation-quality
signals back into detector post-processing thresholds.

The core closed-loop flow is:

```text
RGB image
  -> DetectorWrapper
  -> VLMJudge
  -> XAISelector
  -> XAIExplainer
  -> AutoEvaluator
  -> AdaptiveThreshold + FeedbackLoop
  -> PipelineResult
```

The source tree is the source of truth. Some older README or notebook text may
lag behind implementation details.

## Repository Layout

```text
config/
  detector_config.yaml
  vlm_config.yaml
  xai_config.yaml
  eval_config.yaml

src/
  detector.py
  vlm_judge.py
  xai_selector.py
  evaluator.py
  threshold.py
  feedback_loop.py
  pipeline.py
  utils.py
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
  generate_figures.py

notebooks/
  00_setup_and_data.ipynb
  01_explore_detections.ipynb
  02_vlm_judge_analysis.ipynb
  03_xai_comparison.ipynb
  04_results_visualization.ipynb
  run_full_pipeline.ipynb

tests/
  test_detector.py
  test_vlm_judge.py
  test_xai_methods.py
  test_selector.py
  test_evaluator.py
  test_feedback_loop.py
  test_pipeline.py
  test_threshold.py
```

The import style is currently:

```python
from src.detector import DetectorWrapper
```

Do not rewrite imports to a different package name unless the project packaging
is intentionally changed.

## Runtime Targets

Primary intended runtime:

```text
Linux, Python 3.10+
NVIDIA GPU
CUDA 12.x or newer driver-compatible runtime
24GB VRAM minimum for RTX 3090/4090 class workflows
```

Known target machines:

```text
NVIDIA RTX 4090/4090 Ti, AMD64, 24GB VRAM
NVIDIA DGX Spark / GB10, ARM64, 128GB unified VRAM, CUDA 13.0
```

For DGX Spark ARM64, prefer an NVIDIA PyTorch/NGC container over a bare venv.
Do not let `pip install -r requirements.txt` replace the container-provided
PyTorch build unless the user explicitly requests it.

Recommended DGX Spark pattern:

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

Inside an NVIDIA PyTorch container:

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-build-isolation -e .
grep -vE '^(torch|torchvision)' requirements.txt > /tmp/requirements-no-torch.txt
python -m pip install -r /tmp/requirements-no-torch.txt
python -m pip install --no-build-isolation -r requirements-yolox.txt
```

## Environment Checks

Check GPU and CUDA:

```bash
nvidia-smi
nvcc --version
```

Check Python CUDA stack:

```bash
python - <<'PY'
import platform, torch
print("machine:", platform.machine())
print("torch:", torch.__version__)
print("torch CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("Capability:", torch.cuda.get_device_capability(0))
    print("Device count:", torch.cuda.device_count())
PY
```

Check critical packages:

```bash
python - <<'PY'
mods = ["torch", "torchvision", "bitsandbytes", "transformers", "qwen_vl_utils", "yolox", "pycocotools"]
for m in mods:
    try:
        __import__(m)
        print(f"{m}: OK")
    except Exception as e:
        print(f"{m}: FAIL -> {type(e).__name__}: {e}")
PY
```

## Local Setup

Basic virtualenv setup:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-build-isolation -e .
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation -r requirements-yolox.txt
```

Important: YOLOX must be installed after PyTorch is available because its build
imports `torch`. Use `--no-build-isolation`.

Download data and weights:

```bash
chmod +x scripts/download_data.sh
bash scripts/download_data.sh
```

Expected runtime files:

```text
data/coco/val2017/
data/coco/annotations/instances_val2017.json
data/checkpoints/yolox_s.pth
```

## Core Modules

### `src/detector.py`

Defines:

```python
Detection
DetectorWrapper
compute_scene_complexity
```

Supported detectors:

```text
yolox-s
fasterrcnn_resnet50_fpn_v2
```

Detection outputs are normalized to COCO 80-class IDs.

Important contracts:

```python
detector = DetectorWrapper("yolox-s", config_path="config/detector_config.yaml")
detector.load_model()
detections = detector.detect(image, nms_thresh=None, conf_thresh=None)
model = detector.get_model()
target_layer = detector.get_target_layer()
detector.unload_model()
```

YOLOX preprocessing expects `0..255` float32 image values, not ImageNet
normalization and not division by 255.

### `src/vlm_judge.py`

Defines:

```python
Assessment
VLMJudge
```

Uses Qwen2.5-VL through:

```python
Qwen2VLForConditionalGeneration
AutoProcessor
qwen_vl_utils.process_vision_info
```

Default VLM:

```text
Qwen/Qwen2.5-VL-7B-Instruct
```

Default quantization:

```text
int4
```

On 24GB GPUs use `int4`. On DGX Spark/GB10 with 128GB memory, `fp16` may be a
safer fallback if `bitsandbytes` has ARM64/GB10 issues.

### `src/xai_methods/`

Defines:

```python
SaliencyMap
XAIExplainer
get_explainer(method_name, config)
```

Supported methods:

```text
gradcam
gcame
dclose
lime
```

Contracts:

```python
explainer = get_explainer("gcame", method_config)
saliency = explainer.explain(model, image, detection, target_layer)
assert saliency.map.shape == image.shape[:2]
assert 0.0 <= saliency.map.min()
assert saliency.map.max() <= 1.0
```

CAM methods need `target_layer`. Perturbation methods ignore it.

Runtime notes:

```text
GradCAM: fast, gradient-based
G-CAME: gradient-based with bbox-centered Gaussian weighting
D-CLOSE: slow, segmentation perturbation based
LIME: slow, superpixel perturbation based
```

### `src/xai_selector.py`

Defines:

```python
XAISelectorMLP
XAISelector
SelectorFeatures
METHOD_NAMES
```

Input features:

```text
class_id
confidence
relative_size_encoded
scene_complexity_encoded
num_detections
bbox_aspect_ratio
image_entropy
```

If no trained model exists, selector uses rule-based fallback. Trained model
checkpoint default:

```text
data/checkpoints/xai_selector.pth
```

### `src/evaluator.py`

Defines:

```python
EvalResult
AutoEvaluator
```

Metrics are annotation-free for explanations. Detector boxes serve as pseudo
ground-truth for localization-style metrics.

Metrics:

```text
PG
EBPG
Insertion AUC
Deletion AUC
OA = insertion_auc - deletion_auc
Sparsity / Gini
Composite score
```

Composite score currently uses equal weights:

```text
PG: 1/3
OA: 1/3
Sparsity: 1/3
```

### `src/threshold.py`

Defines:

```python
AdaptiveThreshold
```

Modes:

```text
fixed
percentile
learned
```

Default feedback config uses percentile thresholding.

### `src/feedback_loop.py`

Defines:

```python
FeedbackIteration
FeedbackResult
FeedbackLoop
```

Feedback loop adjusts detector post-processing thresholds based on explanation
quality:

```text
low PG -> decrease NMS threshold
low OA -> increase confidence threshold
```

### `src/pipeline.py`

Defines:

```python
PipelineConfigPaths
PipelineResult
AEDXAIPipeline
```

Primary orchestration entry point:

```python
from src.pipeline import AEDXAIPipeline

pipeline = AEDXAIPipeline(
    detector_config_path="config/detector_config.yaml",
    vlm_config_path="config/vlm_config.yaml",
    xai_config_path="config/xai_config.yaml",
    eval_config_path="config/eval_config.yaml",
)
pipeline.setup()
result = pipeline.run_on_image("data/coco/val2017/000000000139.jpg")
pipeline.shutdown()
```

`setup()` loads detector and VLM. VLM remains resident until `shutdown()` to
avoid repeated 10-20 second reloads during batches.

## Data Contracts

`Detection`:

```python
Detection(
    bbox: list[int],        # [x1, y1, x2, y2]
    class_id: int,          # COCO 0..79 class index
    class_name: str,
    confidence: float,
    area: int,
    relative_size: str,     # "small", "medium", "large"
    detection_id: int,
)
```

`SaliencyMap`:

```python
SaliencyMap(
    map: np.ndarray,        # H x W, float32, [0, 1]
    method_name: str,
    computation_time: float,
    detection_id: int,
)
```

`Assessment`:

```python
Assessment(
    detection_id: int,
    quality_score: float,
    scene_complexity: str,
    object_relative_size: str,
    is_false_positive: bool,
    reasoning: str,
)
```

`EvalResult`:

```python
EvalResult(
    pg: float,
    ebpg: float,
    oa: float,
    insertion_auc: float,
    deletion_auc: float,
    sparsity: float,
    composite: float,
    computation_time: float,
)
```

`PipelineResult`:

```python
PipelineResult(
    image_path: str,
    detections: list[Detection],
    assessments: list[Assessment],
    saliency_maps: list[SaliencyMap],
    evaluation_results: list[EvalResult],
    composite_score: float | None,
    metadata: dict,
)
```

## Config Files

### `config/detector_config.yaml`

Top-level section:

```yaml
detector:
```

Important fields:

```yaml
primary.name: yolox-s
secondary.name: fasterrcnn_resnet50_fpn_v2
conf_thresh: 0.25
nms_thresh: 0.45
device: cuda
```

### `config/vlm_config.yaml`

Top-level section:

```yaml
vlm:
```

Important fields:

```yaml
model_name: Qwen/Qwen2.5-VL-7B-Instruct
quantization: int4
device: cuda
temperature: 0.0
```

Do not pass `temperature=0.0` to `model.generate()` with `do_sample=False`.
The current implementation guards this.

### `config/xai_config.yaml`

Top-level section:

```yaml
xai:
```

Important fields:

```yaml
methods.gradcam.enabled: true
methods.gcame.gaussian_sigma: 0.35
methods.dclose.num_masks_dev: 200
methods.lime.num_perturbations: 500
```

`gcame.gaussian_sigma` is a fraction of bbox scale, not an absolute pixel sigma.

### `config/eval_config.yaml`

Top-level section:

```yaml
evaluation:
```

Important fields:

```yaml
composite_weights:
  pg: 0.333333
  oa: 0.333333
  sparsity: 0.333333
feedback.threshold_mode: percentile
feedback.threshold_percentile: 40
feedback.max_iterations: 3
```

## Scripts

### Download data

```bash
bash scripts/download_data.sh
```

### Train selector

Small smoke run:

```bash
python scripts/train_selector.py \
  --max-images 20 \
  --target-detections 100 \
  --checkpoint-every 25 \
  --oracle-mode
```

Larger run:

```bash
python scripts/train_selector.py \
  --max-images 500 \
  --target-detections 2000 \
  --oracle-mode \
  --resume
```

Selector oracle labeling is expensive because every selected detection is
explained by all configured XAI methods.

### Run full AED-XAI experiments

```bash
python scripts/run_experiments.py \
  --images-dir data/coco/val2017 \
  --num-images 200 \
  --output results/aedxai \
  --selector-model data/checkpoints/xai_selector.pth \
  --seed 42
```

Outputs:

```text
results/aedxai/results.csv
results/aedxai/summary.json
```

### Run fixed-method baseline

```bash
python scripts/run_baseline.py \
  --images-dir data/coco/val2017 \
  --num-images 200 \
  --output results/baseline \
  --xai-method gradcam \
  --seed 42
```

Outputs:

```text
results/baseline/baseline_gradcam_results.csv
```

## Notebooks

Notebook paths are relative to the `notebooks/` directory.

```text
00_setup_and_data.ipynb
```

Local setup, dependency install, COCO download, import checks.

```text
01_explore_detections.ipynb
```

Qualitative YOLOX-S exploration on COCO val2017.

```text
02_vlm_judge_analysis.ipynb
```

VLM quality/complexity/false-positive analysis. Uses cache:

```text
data/vlm_cache_20images.json
```

```text
03_xai_comparison.ipynb
```

Side-by-side XAI comparison. Uses cache:

```text
data/xai_comparison_cache.json
```

```text
04_results_visualization.ipynb
```

Publication figures and ablation tables. Reads:

```text
data/results/*.csv
data/results/*.json
```

If results are absent, synthetic placeholder data is generated.

```text
run_full_pipeline.ipynb
```

Interactive end-to-end run, small selector training, AED-XAI vs GradCAM
baseline comparison, and saved outputs.

## Testing And Validation

Fast syntax checks:

```bash
python -m compileall src scripts tests
```

Targeted unit tests:

```bash
pytest tests/test_threshold.py
pytest tests/test_selector.py
pytest tests/test_evaluator.py
pytest tests/test_feedback_loop.py
```

Model-dependent tests:

```bash
pytest tests/test_detector.py
pytest tests/test_xai_methods.py
pytest tests/test_vlm_judge.py -m "not slow"
```

Slow/GPU tests:

```bash
pytest -m slow
```

Before handing back code changes, at minimum run:

```bash
python -m compileall src scripts tests
```

For notebook-only edits, validate JSON:

```bash
python -m json.tool notebooks/<name>.ipynb >/tmp/notebook.json
```

Optionally compile notebook code cells:

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path("notebooks/run_full_pipeline.ipynb")
nb = json.loads(p.read_text())
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        compile("".join(cell["source"]), f"cell_{i}", "exec")
print("OK")
PY
```

## Coding Rules For Agents

- Prefer small, targeted edits.
- Use `pathlib.Path` for new Python file I/O.
- Use context managers for file reads/writes.
- Keep heavy imports lazy where existing modules already do that.
- Preserve CPU fallback where possible, but VLM CUDA-only behavior is acceptable
  when config requests CUDA.
- Do not hard-code absolute paths.
- Do not commit or add large files from `data/`, `results/`, checkpoints, or
  Hugging Face caches.
- Do not change metric definitions or composite weights casually; those are
  paper-facing research choices.
- Do not weaken tests to pass local CPU timing unless the test is explicitly
  marked/skipped for CPU-only environments.
- Do not silently change notebook kernels unless the user asks about notebook
  environment issues.

## Known Sharp Edges

- YOLOX is installed from GitHub and can fail if build isolation hides `torch`.
  Use:

  ```bash
  python -m pip install --no-build-isolation -r requirements-yolox.txt
  ```

- On macOS, GPU/CUDA workflows will not run. Use Linux/NVIDIA hosts for actual
  pipeline experiments.
- `bitsandbytes` can be platform-sensitive on ARM64/GB10. If int4 VLM loading
  fails on DGX Spark, switch `config/vlm_config.yaml` to:

  ```yaml
  quantization: "fp16"
  ```

- D-CLOSE and LIME are intentionally slow. Use small image counts for smoke
  tests.
- `AEDXAIPipeline.setup()` loads the VLM and keeps it resident until
  `shutdown()`. This is intentional for batch speed.
- Notebook output can be huge. Keep committed notebooks clean unless the user
  specifically asks for executed outputs.
- `data/` and `results/` are runtime/output directories and should usually stay
  untracked.

## Quick Smoke Commands

Detector smoke:

```bash
python - <<'PY'
from pathlib import Path
from src.detector import DetectorWrapper
from src.utils import load_image

image_path = Path("data/coco/val2017/000000000139.jpg")
detector = DetectorWrapper("yolox-s", config_path="config/detector_config.yaml")
detector.load_model()
image = load_image(str(image_path))
detections = detector.detect(image)
print("detections:", len(detections))
print(detections[:3])
detector.unload_model()
PY
```

VLM load smoke:

```bash
python - <<'PY'
from src.vlm_judge import VLMJudge
judge = VLMJudge(config_path="config/vlm_config.yaml")
judge.load_model()
print("VLM loaded")
judge.unload_model()
PY
```

Pipeline one-image smoke:

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
print(result.composite_score, result.metadata)
pipeline.shutdown()
PY
```

## Research Framing

Use this framing when adding documentation, tables, or paper-facing outputs:

- AED-XAI does **not** require human explanation annotations.
- Detector boxes are used as pseudo-ground-truth for explanation localization
  metrics.
- The XAI selector is trained offline using oracle labels generated by running
  all candidate XAI methods and selecting the method with the highest composite
  score.
- At inference, the selector predicts from pre-explanation context features
  only.
- Adaptive thresholding makes feedback decisions robust to score distribution
  shifts across domains.
