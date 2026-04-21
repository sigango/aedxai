# AED-XAI Agent Guide

This file is the working context map for AI coding agents and collaborators. Read it before changing code. It records the current implementation, research contracts, runtime assumptions, and safe validation commands for AED-XAI.

## Project Mission

AED-XAI stands for **Adaptive Explanation-Driven Object Detection with XAI**.

The pipeline:

```text
image
  -> object detector
  -> VLM detection judge
  -> adaptive XAI method selector
  -> explanation method
  -> annotation-free evaluator
  -> adaptive threshold + feedback loop
  -> pipeline result
```

The key research claim is that object-detection explanations should be selected per detection, not fixed globally. The selector is trained offline from oracle labels produced by exhaustive XAI evaluation, then predicts online from pre-explanation context features only.

## Current Source Of Truth

Use the source tree and config files as the source of truth. Some notebooks may be explanatory or cached, but source modules and scripts define behavior.

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
  pipeline_io.py
  utils.py
  xai_methods/

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

The project currently imports modules as:

```python
from src.detector import DetectorWrapper
```

Do not rename the package or rewrite imports unless the user explicitly asks for a packaging refactor.

## Runtime Targets

Primary runtime:

```text
Linux
Python 3.10+
NVIDIA GPU
CUDA 12.x or newer driver-compatible runtime
24GB VRAM minimum for full RTX 3090/4090 workflows
```

Known target machines:

```text
NVIDIA RTX 4090/4090 Ti, AMD64, 24GB VRAM
NVIDIA DGX Spark / GB10, ARM64, 128GB unified VRAM, CUDA 13.0
```

On DGX Spark/ARM64, prefer an NVIDIA PyTorch/NGC container. Do not blindly `pip install torch torchvision` over the container builds. If `bitsandbytes` is unstable on ARM64/GB10, change `config/vlm_config.yaml` to:

```yaml
quantization: "fp16"
```

## Environment Setup Commands

Container-first setup:

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

Virtualenv setup:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-build-isolation -e .
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation -r requirements-yolox.txt
```

YOLOX must be installed after PyTorch is importable. Use `--no-build-isolation`.

## Cluster Checks

GPU/CUDA:

```bash
nvidia-smi
nvcc --version
```

Python CUDA stack:

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
    print("VRAM GB:", torch.cuda.get_device_properties(0).total_memory / 1e9)
PY
```

Critical imports:

```bash
python - <<'PY'
mods = [
    "torch",
    "torchvision",
    "transformers",
    "qwen_vl_utils",
    "bitsandbytes",
    "yolox",
    "pycocotools",
    "cv2",
    "captum",
]
for name in mods:
    try:
        __import__(name)
        print(f"{name}: OK")
    except Exception as exc:
        print(f"{name}: FAIL -> {type(exc).__name__}: {exc}")
PY
```

## Core Data Contracts

### `Detection`

Defined in `src/detector.py`.

```python
Detection(
    bbox: list[int],        # [x1, y1, x2, y2]
    class_id: int,          # COCO 0..79
    class_name: str,
    confidence: float,
    area: int,
    relative_size: str,     # "small", "medium", "large"
    detection_id: int,
)
```

### `Assessment`

Defined in `src/vlm_judge.py`.

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

### `SaliencyMap`

Defined in `src/xai_methods/base.py`.

```python
SaliencyMap(
    map: np.ndarray,        # H x W float32, [0, 1]
    method_name: str,
    computation_time: float,
    detection_id: int,
)
```

### `EvalResult`

Defined in `src/evaluator.py`.

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

### `PipelineResult`

Defined in `src/pipeline.py`.

```python
PipelineResult(
    image_path: str,
    detections: list[Detection],
    assessments: list[Assessment],
    saliency_maps: list[SaliencyMap],
    evaluation_results: list[EvalResult],
    composite_score: float | None,
    metadata: dict,
    selector_reasoning: dict[int, dict],
)
```

## Core Modules

### `src/detector.py`

Public API:

```python
DetectorWrapper
Detection
compute_scene_complexity
```

Supported detectors:

```text
yolox-s
fasterrcnn_resnet50_fpn_v2
```

Important notes:

- Detection outputs are normalized to COCO 80-class IDs.
- YOLOX preprocessing expects `0..255` float32 image values.
- Do not divide YOLOX fallback preprocessing by 255.
- `get_target_layer()` must return a real `torch.nn.Module`, not a string.
- NMS re-application is intentional because the feedback loop controls NMS.

### `src/vlm_judge.py`

Public API:

```python
VLMJudge
Assessment
```

Uses:

```python
Qwen2VLForConditionalGeneration
AutoProcessor
qwen_vl_utils.process_vision_info
```

Generation must not pass `temperature=0.0` with `do_sample=False`. The current code only passes sampling parameters when `temperature > 0.0`.

### `src/xai_methods/`

Public API:

```python
XAIExplainer
SaliencyMap
get_explainer(method_name, config)
```

Methods:

```text
gradcam
gcame
dclose
lime
```

Important notes:

- CAM methods need `target_layer`.
- D-CLOSE and LIME ignore `target_layer`.
- G-CAME `gaussian_sigma` is a bbox-scale fraction. Default is `0.35`.
- D-CLOSE and LIME use neutral fill value `114` for masked regions.
- LIME kernel distance is normalized by number of superpixels.

### `src/xai_selector.py`

Public API:

```python
XAISelectorMLP
XAISelector
SelectorFeatures
METHOD_NAMES
METHOD_TO_IDX
```

Selector features:

```text
class_id
confidence
relative_size_encoded
scene_complexity_encoded
num_detections
bbox_aspect_ratio
image_entropy
```

Class ID is normalized to `[0, 1]` by division by `79.0` before MLP inference. Keep `SelectorFeatures.class_id` as an integer.

Checkpoint behavior:

```text
data/checkpoints/xai_selector_<detector>.pth
data/checkpoints/xai_selector.pth       # legacy fallback
```

If no trained checkpoint exists, the selector uses rule-based fallback.

### `src/evaluator.py`

Public API:

```python
AutoEvaluator
EvalResult
EvaluationResult  # alias
```

Metrics are annotation-free. Detector boxes are pseudo-ground-truth.

Composite score:

```text
EBPG      1/3
OA        1/3, normalized to [0, 1]
Sparsity  1/3
```

Do not casually change composite definitions or weights. They are paper-facing.

### `src/threshold.py`

Public API:

```python
AdaptiveThreshold
```

Modes:

```text
fixed
percentile
learned
```

Default config uses:

```yaml
threshold_mode: "percentile"
threshold_percentile: 40
```

### `src/feedback_loop.py`

Public API:

```python
FeedbackLoop
FeedbackResult
FeedbackIteration
```

Feedback behavior:

```text
low EBPG -> decrease NMS threshold
low OA   -> increase confidence threshold
```

The loop stores VLM assessments and selector reasoning in `FeedbackResult`.

### `src/pipeline.py`

Public API:

```python
AEDXAIPipeline
PipelineResult
PipelineConfigPaths
```

Important behavior:

- `setup()` loads detector and VLM.
- VLM remains resident until `shutdown()` to avoid repeated 10-20 second reloads.
- Detector model comes from `detector.primary.name` unless `detector_model_name` override is passed.
- Selector checkpoint is chosen per detector when available.

### `src/pipeline_io.py`

Public API:

```python
save_pipeline_result(result, output_dir, save_saliency_npy=False)
```

Writes:

```text
result.json
detections.png
det_000_saliency.png
det_000_saliency.npy   # optional
```

## Config Contracts

### `config/detector_config.yaml`

Top-level key:

```yaml
detector:
```

Important fields:

```yaml
primary.name: "yolox-s"
secondary.name: "fasterrcnn_resnet50_fpn_v2"
conf_thresh: 0.25
nms_thresh: 0.45
device: "cuda"
```

### `config/vlm_config.yaml`

Top-level key:

```yaml
vlm:
```

Important fields:

```yaml
model_name: "Qwen/Qwen2.5-VL-7B-Instruct"
quantization: "int4"
temperature: 0.0
device: "cuda"
```

### `config/xai_config.yaml`

Top-level key:

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

### `config/eval_config.yaml`

Top-level key:

```yaml
evaluation:
```

Important fields:

```yaml
composite_weights:
  ebpg: 0.333333
  oa: 0.333333
  sparsity: 0.333333
feedback.threshold_mode: "percentile"
feedback.threshold_percentile: 40
feedback.max_iterations: 3
```

## Script Guide

### Download Data

```bash
bash scripts/download_data.sh
```

### Train Selector

```bash
python scripts/train_selector.py \
  --detector-model yolox-s \
  --max-images 20 \
  --target-detections 100 \
  --checkpoint-every 25 \
  --oracle-mode
```

Detector-specific larger runs:

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

Oracle labeling protocol:

1. Run all configured XAI methods for each selected detection.
2. Evaluate each method with annotation-free metrics.
3. Normalize EBPG, OA, and Sparsity across the dataset.
4. Assign the oracle label as the method with highest equal-weight composite.
5. Train the MLP from pre-explanation features only.

### Run Full Pipeline Experiments

```bash
python scripts/run_experiments.py \
  --images-dir data/coco/val2017 \
  --num-images 3000 \
  --output results/aedxai \
  --detector-model yolox-s \
  --seed 42
```

### Run Fixed Baseline

```bash
python scripts/run_baseline.py \
  --images-dir data/coco/val2017 \
  --num-images 3000 \
  --output results/baseline \
  --detector-model yolox-s \
  --xai-method gradcam \
  --seed 42
```

### Compare Detectors

```bash
python scripts/compare_detectors.py \
  --max-images 3000 \
  --detectors yolox-s,fasterrcnn_resnet50_fpn_v2
```

### Cross-Domain Evaluation

```bash
python scripts/run_cross_domain.py \
  --num-images 200 \
  --methods aedxai,gradcam,gcame,dclose,lime \
  --output-dir results \
  --recursive
```

Default domains are COCO, VOC, BDD100K, VisDrone, DOTA, and OpenImages.
Missing folders are skipped unless `--strict` is passed. The output consumed
by notebook 05 is `results/cross_domain.csv`.

### Selector-Size Ablation

```bash
python scripts/ablation_selector_size.py \
  --training-csv results/xai_selector_training_data_yolox-s.csv \
  --sizes 50,100,200,500,1000 \
  --seeds 42,123,456
```

Notebook 04 runs selector-size ablations for both detector-specific training
CSVs and notebook 05 plots the detector-specific curves.

### Generate Figures

```bash
python scripts/generate_figures.py \
  --results-root results \
  --figures-dir results/figures
```

## Notebook Guide

Notebook paths are relative to `notebooks/`.

```text
00_setup_and_data.ipynb       setup, install, data download
01_explore_detections.ipynb   YOLOX qualitative detection exploration
02_vlm_judge_analysis.ipynb   VLM quality and complexity analysis
03_xai_comparison.ipynb       XAI method comparison
04_run_all_experiments.ipynb  full experiment runner, selector training, 3000-image evaluation, exports
05_results_visualization.ipynb figures and ablation tables from saved outputs
```

Keep committed notebooks clean unless the user asks for executed outputs. Expensive notebooks use caches under `data/`.

## Validation Commands

Always run syntax checks after code edits:

```bash
python -m compileall src scripts tests
```

Fast unit tests:

```bash
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

Notebook JSON validation:

```bash
python -m json.tool notebooks/<name>.ipynb >/tmp/notebook.json
```

Compile notebook code cells:

```bash
python - <<'PY'
import ast, json
from pathlib import Path

for path in sorted(Path("notebooks").glob("*.ipynb")):
    nb = json.loads(path.read_text())
    for index, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") == "code":
            ast.parse("".join(cell.get("source", [])) or "\n", filename=f"{path}:cell{index}")
print("Notebook code cells parse OK")
PY
```

## Coding Rules For Agents

- Prefer small, targeted edits.
- Use `pathlib.Path` for new Python file I/O.
- Use context managers for file reads and writes.
- Keep heavy imports lazy where existing modules already do that.
- Preserve CPU importability where feasible, but full pipeline execution is GPU-oriented.
- Do not hard-code absolute paths.
- Do not commit runtime data from `data/`, `results/`, checkpoints, Hugging Face caches, or large notebook outputs.
- Do not weaken timing or GPU tests to pass local CPU-only environments.
- Do not change metric definitions, threshold logic, or composite weights casually.
- Do not change VLM generation semantics without checking transformers compatibility.
- Do not replace container-provided PyTorch on ARM64 unless the user explicitly asks.

## Sharp Edges

- YOLOX build can fail if build isolation hides `torch`.
- `bitsandbytes` may be platform-sensitive on ARM64/GB10.
- D-CLOSE and LIME are intentionally slow.
- VLM loading is slow, so `AEDXAIPipeline` keeps it resident until `shutdown()`.
- Local macOS environments cannot validate CUDA behavior.
- Notebook kernels often point at the wrong Python environment.
- `results/` and `data/` are usually gitignored runtime directories.

## Smoke Commands

Detector:

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

VLM:

```bash
python - <<'PY'
from src.vlm_judge import VLMJudge

judge = VLMJudge(config_path="config/vlm_config.yaml")
judge.load_model()
print("VLM loaded")
judge.unload_model()
PY
```

Full pipeline:

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

Use this language in docs and paper-facing outputs:

- AED-XAI does not require human explanation annotations.
- Detector boxes are used as pseudo-ground-truth for explanation localization metrics.
- The selector is trained offline using oracle labels generated by exhaustive XAI evaluation.
- At inference, the selector predicts from pre-explanation context features only.
- Adaptive thresholding makes feedback decisions robust to score distribution shifts across domains.
- Percentile thresholding is the out-of-the-box adaptive mode; learned thresholding is a calibration baseline.
- Composite quality uses equal weights over EBPG, normalized OA, and Sparsity.
