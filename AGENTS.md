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
  evidence_config.yaml        # v2 evidence runs (full, dev pilot)
  evidence_config_lite.yaml   # v2 TEST config (COCO test, BDD100K): full method list, <=10 explained/image

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
  gt_matching.py      # v2: COCO-format GT, TP/FP matching, AP
  evidence.py         # v2: per-detection evidence features
  evidence_io.py      # v2: shard records -> per-detection DataFrame (CPU)
  provenance.py       # run manifests, image manifests
  notebook_utils.py   # v2 notebooks: run tasks/stages, progress, budget, tau, gates
  v2_figures.py       # v2 paper figures (matplotlib, lazy import)
  xai_methods/        # incl. controls.py (bbox/center prior, random, shuffle)

scripts/
  download_data.sh
  train_selector.py
  run_experiments.py
  run_baseline.py
  compare_detectors.py
  run_cross_domain.py
  ablation_selector_size.py
  generate_figures.py
  make_manifests.py            # v2: disjoint dev/calib/test image manifests
  convert_annotations.py       # v2: VOC/BDD100K/VisDrone -> COCO-80 JSON
  run_evidence.py              # v2: GPU stage 1 (sharded, resumable)
  run_vlm_verification.py      # v2: GPU stage 2
  analyze_evidence.py          # v2: gates G1-G3, abstention (CPU)
  analyze_metric_validity.py   # v2: metric audit, headroom (CPU)
  calibrate_tau.py             # per-detector cascade tau

cluster/                        # task.sh, submit.sh, env_check.sh, cluster.env.example
manifests/                      # committed image manifests (created by make_manifests.py)

notebooks/
  00_setup_and_data.ipynb
  01_explore_detections.ipynb
  02_vlm_judge_analysis.ipynb
  03_xai_comparison.ipynb
  04_run_all_experiments.ipynb
  05_results_visualization.ipynb
  06_v2_setup_and_smoke.ipynb
  07_v2_run_experiments.ipynb
  08_v2_analysis_and_figures.ipynb

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
    is_fallback: bool = False,         # True = bbox-prior substituted for a failed/zero map
    fallback_reason: str | None = None,
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
gcame_raw      # variant: use_bbox_center_weighting=false
gradcam_raw    # variant: renormalize_in_bbox=false
bbox_prior     # control: Gaussian on the detection box
center_prior   # control: Gaussian on the image centre
random         # control: seeded uniform noise
```

`shuffle_saliency(map, seed)` builds the pixel-shuffled control. `CAM_METHODS`
lists methods that need a target layer; `base_method_name()` maps a variant to
its `xai_config.yaml` section. The selector's `METHOD_NAMES` stay the four
original methods.

Important notes:

- CAM methods need `target_layer`.
- D-CLOSE and LIME ignore `target_layer`.
- G-CAME `gaussian_sigma` is a bbox-scale fraction. Default is `0.35`.
- D-CLOSE and LIME use neutral fill value `114` for masked regions.
- LIME kernel distance is normalized by number of superpixels.
- LIME and D-CLOSE batch their detector forwards with the method's `batch_size`
  (`_forward_detector_batch`); batch size changes speed only, not the map.
- Explainers set `SaliencyMap.is_fallback` when the bbox prior replaces a failed map.

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

`insertion_deletion` is vectorised and returns the same values as the original
per-pixel loop (see `tests/test_evaluator_curves.py`); `metrics.over_all.batch_size`
> 1 batches the perturbation forwards without changing values.

`evaluate_evidence()` (v2 experiments only) returns all `evaluate_all` metrics
plus the insertion/deletion curves, the box-relative protocol (`*_box`: pixels
ranked and perturbed only inside the detection box scaled by
`metrics.over_all_box.context_scale`, everything else kept original) and
`pg_gt`/`ebpg_gt` for an annotated box. `composite_box` is a sensitivity column,
never the paper-facing composite.

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

Evaluate-then-escalate cascade (config `feedback.escalation`, **off by
default**): when a detection's composite falls below the detector's `tau`
(`escalation.tau_by_detector[detector.model_name]`, falling back to
`escalation.tau`), the backup method runs and the better-scoring map is kept
(keep-max never decreases the selection metric for the same detections).
Calibrate `tau_by_detector` with `scripts/calibrate_tau.py` on a manifest
split disjoint from every test split, and freeze it in config — never compute
it from test images. Recalibrate whenever `composite_weights` change.

`feedback.fixed_method` bypasses the selector (e.g. `gcame` + escalation gives
the always-G-CAME -> backup cascade). `num_escalations` counts the returned
iteration only.

Optional `feedback.vlm_fp_filter` skips explaining VLM-flagged false
positives; flags from iteration 0 are re-applied to re-detections by class and
IoU. `run_baseline.py` has no VLM, so an FP-filtered pipeline run is not
comparable to fixed baselines; in v2 the VLM is an evidence feature instead.

The loop stores VLM assessments, selector reasoning, and `num_escalations`
in `FeedbackResult`.

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
- `feedback_overrides` is deep-merged over `eval_config.yaml: evaluation.feedback`
  (`run_experiments.py --max-iterations/--fixed-method/--escalation`).
- `use_vlm=False` (`--no-vlm`) skips loading the VLM; scene complexity then comes
  from detector statistics, which is also the selector's training feature.

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
metrics.over_all.batch_size: 1          # >1 = batched forwards, same values
metrics.over_all_box.context_scale: 2.0  # box-relative protocol (evaluate_evidence only)
feedback.vlm_fp_filter: false
feedback.fixed_method: null
feedback.escalation.enabled: false
feedback.escalation.backup_method: "lime"
feedback.escalation.tau_by_detector:     # provisional until calibrate_tau.py on the calib split
  yolox-s: 0.6546
  fasterrcnn_resnet50_fpn_v2: 0.6922
```

### `config/evidence_config.yaml` / `evidence_config_lite.yaml`

Top-level keys `evidence` (methods, controls, shuffled maps, agreement pairs,
gradient features, `prenms` and `tta` strong baselines, `max_explained_per_image`,
saved maps), `vlm_verification`
and `analysis` (CV folds, bootstrap, gate `gate_delta_auroc: 0.02`, conformal
`crc_alpha`, acquisition budgets, and the pre-registered `claims` /
`audit_claims` that `analyze_evidence.py` / `analyze_metric_validity.py` turn
into SUPPORTED / NOT SUPPORTED verdicts). The test config, claims and image caps
are frozen with tag `v2-freeze-test` (`PREREGISTRATION.md`, scenario C after G1
failed on the dev pilot); do not edit them before the test and BDD100K analyses
have run.

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
3. Compute each method's composite on RAW metric values exactly as
   `AutoEvaluator.composite_score` does, with weights read from
   `config/eval_config.yaml:composite_weights` (single source of truth —
   the training target always equals the reported metric).
4. Assign the oracle label as the argmax composite; drop detections whose
   top-1 vs top-2 margin is below 0.02 (raw scale) as ambiguous.
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

Synthetic placeholder data is off by default (`--allow-synthetic` to enable;
never for paper figures).

### v2 Explanation-As-Evidence Runs

`RESULTS_AND_PLAN.md` (Part B) is the runbook; every job is defined once in `cluster/task.sh`
and launched with `cluster/submit.sh <stage>` (SLURM arrays or local GPUs).

```bash
python scripts/make_manifests.py --images-dir data/coco/val2017 \
  --annotations data/coco/annotations/instances_val2017.json \
  --splits dev:300,calib:1000,test:2000 --output manifests/coco_val2017_v2.json

python scripts/run_evidence.py --images-dir data/coco/val2017 \
  --annotations data/coco/annotations/instances_val2017.json \
  --image-list manifests/coco_val2017_v2.json --split dev \
  --detector-model yolox-s --run-name coco_dev_yolox-s --num-shards 4 --shard-index 0

python scripts/run_vlm_verification.py --run-dir results/evidence/coco_dev_yolox-s \
  --images-dir data/coco/val2017

python scripts/analyze_evidence.py --run-dirs results/evidence/coco_dev_yolox-s \
  --output-dir results/analysis/evidence_coco_dev
python scripts/analyze_metric_validity.py --run-dirs results/evidence/coco_dev_yolox-s \
  --output-dir results/analysis/metric_validity_coco_dev
```

Evidence features never use GT-derived columns (`*_gt`, `iou_gt`, labels); GT
is only the TP/FP endpoint. The G1 gate is `S3_explanation - S2_baseline`, where
`S2_baseline` = post-NMS output (`out__`) + pre-NMS candidates
(`DetectorWrapper.detect_candidates`, `pre__`) + gradients incl. Riedlinger-style
head-group norms (`grad__`, `grad__rg_*`) + TTA stability (`tta__`); the original
lite gate `S3_lite - S2_output_gradient` is secondary. `cost__*` columns are
costs, never features.

On the shared DGX Spark, start long chains with
`cluster/launch_unit.sh UNIT TASK[:SHARDS] ...` (a systemd user unit running
`cluster/run_chain.sh`: retries, memory log in `logs/UNIT_mem.log`), never from
tmux or a Jupyter terminal. `cluster/task.sh setup-bdd-manifest` builds only the
BDD100K manifest (`setup-manifests` would rewrite the committed COCO manifest). `run_experiments.py` and `run_baseline.py` accept
`--image-list/--split` and write `manifest.json`; `run_experiments.py` also
writes `detections.csv` (chosen method, escalation outcome, fallback flag).

## Notebook Guide

Notebook paths are relative to `notebooks/`.

```text
00_setup_and_data.ipynb       setup, install, data download
01_explore_detections.ipynb   YOLOX qualitative detection exploration
02_vlm_judge_analysis.ipynb   VLM quality and complexity analysis
03_xai_comparison.ipynb       XAI method comparison
04_run_all_experiments.ipynb  v1 (legacy) experiment runner, selector training, exports
05_results_visualization.ipynb v1 (legacy) figures and ablation tables
06_v2_setup_and_smoke.ipynb   v2: env/data/manifests, 3-image smoke run, real-model checks, budget
07_v2_run_experiments.ipynb   v2: stages 1-5 via cluster/submit.sh, D1 decisions, tau, freeze check
08_v2_analysis_and_figures.ipynb v2: gates B/G1-G3, abstention, subgroups, qualitative, v1 rows (CPU)
```

v2 notebooks never define experiment commands: they call `cluster/task.sh` /
`cluster/submit.sh` through `src/notebook_utils.py`, so a notebook run and a
cluster run execute the same commands. Only notebook 06 section 5 runs model
code in the kernel (needs a GPU kernel). Commit notebooks without outputs.

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

v2 tests (CPU, toy detector in `tests/toy_models.py`):

```bash
pytest tests/test_evaluator_curves.py tests/test_gt_matching.py tests/test_controls.py \
  tests/test_evidence_features.py tests/test_evidence_pipeline.py tests/test_batched_explainers.py \
  tests/test_provenance.py tests/test_notebook_helpers.py -q
```

Everything that does not need a GPU or model download:

```bash
pytest -m "not slow" tests -q
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
- v2: explanations are used as evidence to verify detections; verification is
  scored against annotated boxes (TP/FP, AP), while the explanation metrics
  themselves stay annotation-free. Do not describe the evidence model as
  annotation-free during development: it is supervised by GT labels offline and
  annotation-free only at inference.
