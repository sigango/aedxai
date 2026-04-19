# AED-XAI Agent Guide

This file is the fast context map for AI coding agents and humans who need to
work on this repository quickly. Prefer this over guessing from notebooks.

## Project In One Minute

AED-XAI is an object-detection explainability research pipeline. The intended
flow is:

```text
image
  -> DetectorWrapper
  -> VLMJudge
  -> XAISelector
  -> XAI explainer
  -> AutoEvaluator
  -> FeedbackLoop
  -> PipelineResult
```

Core import style is currently `from src...`, not `from aed_xai...`.

The README still contains some older "Stage 1 / TODO skeleton" wording. The
source tree is a better source of truth: detector, VLM judging, XAI methods,
selector, evaluator, thresholding, feedback, and pipeline orchestration all have
real implementations.

## Setup Commands

Use Python 3.10+ with a virtualenv. The local `.venv` may already exist.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install --no-build-isolation -r requirements-yolox.txt
pip install -e .
```

YOLOX is installed from GitHub and must be installed after `requirements.txt`
because its build imports `torch`. Use `--no-build-isolation` for that install.

To download default data/checkpoints:

```bash
bash scripts/download_data.sh
```

Quick syntax check:

```bash
python -m compileall src scripts tests
```

Targeted tests:

```bash
pytest tests/test_threshold.py tests/test_selector.py tests/test_evaluator.py
```

Be careful with full `pytest`: some tests load Faster R-CNN, may download model
weights, and can be slow or GPU-sensitive. YOLOX tests skip if YOLOX is missing.

## Repository Map

- `src/detector.py`: unified detector wrapper for YOLOX-S and torchvision
  Faster R-CNN. Defines `Detection`, `DetectorWrapper`, and scene complexity.
- `src/vlm_judge.py`: Qwen2.5-VL based detection quality assessor. Handles
  image annotation, JSON parsing, retries, and `Assessment` objects.
- `src/xai_methods/`: explanation implementations.
  - `gradcam.py`: Captum LayerGradCam wrapper.
  - `gcame.py`: GradCAM-style method with bbox-centered weighting.
  - `dclose.py`: perturbation/segmentation based D-CLOSE.
  - `lime_det.py`: detection-adapted LIME.
  - `base.py`: shared detector forwarding, target matching, saliency contracts.
- `src/xai_selector.py`: method-selection features, rule fallback, MLP training
  and inference helpers.
- `src/evaluator.py`: automatic explanation metrics and composite scoring.
- `src/threshold.py`: fixed, percentile, and learned adaptive thresholds.
- `src/feedback_loop.py`: iterates detection/explanation/evaluation and adjusts
  detector thresholds when explanation quality is low.
- `src/pipeline.py`: high-level coordinator. Main class is `AEDXAIPipeline`.
- `src/utils.py`: image I/O, COCO classes, geometry, seed/device helpers.
- `scripts/train_selector.py`: generates oracle selector data and trains MLP.
- `scripts/download_data.sh`: downloads COCO sample assets and YOLOX weights.
- `notebooks/`: exploratory workflows. Treat notebook outputs as non-source
  unless the user specifically asks to update them.

## Config Files

- `config/detector_config.yaml`
  - Top-level section: `detector`.
  - Primary detector defaults to `yolox-s`.
  - Secondary detector is `fasterrcnn_resnet50_fpn_v2`.
  - Includes thresholds, max detections, CUDA preference, and scene-complexity
    thresholds.
- `config/vlm_config.yaml`
  - Top-level section: `vlm`.
  - Default model is `Qwen/Qwen2.5-VL-7B-Instruct`, int4 quantized, CUDA.
  - Contains prompt templates and box annotation styling.
- `config/xai_config.yaml`
  - Top-level section: `xai`.
  - Enables `gradcam`, `gcame`, `dclose`, and `lime`.
  - Contains target layer names and per-method runtime settings.
- `config/eval_config.yaml`
  - Top-level section: `evaluation`.
  - Defines metrics, composite weights, feedback thresholds, and experiments.

## Important Data Contracts

`Detection` from `src/detector.py`:

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

`SaliencyMap` from `src/xai_methods/base.py` carries the normalized explanation
map, method name, computation time, and detection id.

`Assessment` from `src/vlm_judge.py` carries VLM quality score, scene
complexity, relative size, false-positive flag, and reasoning.

`EvalResult` from `src/evaluator.py` carries metric outputs and the composite
score used by feedback.

`PipelineResult` from `src/pipeline.py` is the final per-image bundle.

## Common Workflows

Run full pipeline programmatically:

```python
from src.pipeline import AEDXAIPipeline

pipeline = AEDXAIPipeline(
    detector_config_path="config/detector_config.yaml",
    vlm_config_path="config/vlm_config.yaml",
    xai_config_path="config/xai_config.yaml",
    eval_config_path="config/eval_config.yaml",
)

result = pipeline.run_on_image("data/coco/val2017/000000000139.jpg")
print(result.composite_score)
pipeline.shutdown()
```

Use Faster R-CNN for local detector debugging when YOLOX is not installed:

```python
from src.detector import DetectorWrapper
from src.utils import load_image

detector = DetectorWrapper("fasterrcnn_resnet50_fpn_v2")
detector.load_model()
image = load_image("tests/fixtures/000000000139.jpg")
detections = detector.detect(image)
```

Train selector:

```bash
python scripts/train_selector.py --help
```

The selector training path can be expensive because oracle labeling evaluates
all configured XAI methods per selected detection.

## Coding Notes

- Keep imports consistent with the existing `src.*` package layout.
- Prefer YAML config changes over hard-coded constants when tuning model,
  threshold, metric, or XAI method behavior.
- Preserve lazy imports for heavy libraries (`torch`, `transformers`, YOLOX)
  where the code already uses them. This keeps parse-only tests usable.
- Do not assume CUDA is available. Many components prefer CUDA, but tests should
  skip or degrade gracefully where possible.
- XAI methods should return normalized maps in `[0, 1]` with the same height and
  width as the input image unless config says otherwise.
- Detector outputs should stay normalized to the `Detection` dataclass and COCO
  80-class indexing.
- VLM output parsing should tolerate imperfect JSON. Existing repair/retry logic
  lives in `src/vlm_judge.py`.
- Feedback should adjust detector thresholds through existing detector methods
  instead of mutating random attributes.

## Known Sharp Edges

- YOLOX is not a normal PyPI dependency here. Install it with:

  ```bash
  pip install --no-build-isolation -r requirements-yolox.txt
  ```

- Full model tests can download weights and take a while.
- The current git worktree may contain notebook execution changes. Avoid
  rewriting notebooks unless the user asks.
- `data/` and `results/` are runtime/output areas. Avoid committing large model
  checkpoints or generated experiment artifacts unless explicitly requested.
- The package metadata is minimal (`setup.py` only). There is no `pyproject.toml`
  yet.

## Best First Checks After Editing

Run these before handing work back:

```bash
python -m compileall src scripts tests
pytest tests/test_threshold.py tests/test_selector.py tests/test_evaluator.py
```

For detector/XAI changes, also run the most relevant targeted test file, but
expect model-loading cost:

```bash
pytest tests/test_detector.py
pytest tests/test_xai_methods.py
```
