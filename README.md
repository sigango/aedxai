# AED-XAI: Auto-Explainable Object Detection Pipeline

AED-XAI is a research scaffold for a closed-loop object detection interpretability pipeline. The system detects objects, asks a vision-language model to judge detection quality and scene characteristics, adaptively selects the most suitable explanation method, evaluates explanation quality without human annotations, and feeds poor explanation scores back into the detector to refine future predictions. This repository is organized as a Stage 1 project scaffold so each subsystem can be implemented, benchmarked, and ablated independently.

At this stage, the project structure, configs, utilities, fixtures, and entry points are in place, while the core research components intentionally remain TODO skeletons that raise `NotImplementedError("TODO: Stage 2")`.

## Pipeline Overview

```text
              +------------------+
              |   Input Image    |
              +---------+--------+
                        |
                        v
              +------------------+
              | Object Detector  |
              +---------+--------+
                        |
                        v
              +------------------+
              |   VLM Judge      |
              | quality/context  |
              +---------+--------+
                        |
                        v
              +------------------+
              | Adaptive XAI     |
              |    Selector      |
              +---------+--------+
                        |
                        v
              +------------------+
              | XAI Explainer    |
              | GradCAM/GCAME/   |
              | D-CLOSE/LIME     |
              +---------+--------+
                        |
                        v
              +------------------+
              | Explanation Eval |
              | no human labels  |
              +---------+--------+
                        |
             low score? | yes
                        v
              +------------------+
              | Feedback Loop    |
              | refine detector  |
              +---------+--------+
                        |
                        +----> iterate / stop
```

## Installation

### Local Pip Workflow

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install --no-build-isolation -r requirements-yolox.txt
pip install -e .
```

### Docker Workflow

```bash
docker build -t aed-xai .
docker run --gpus all -it --rm -v "$(pwd)":/workspace/aed-xai aed-xai bash
```

## Quick Start

Download the default validation assets and install the package:

```bash
bash scripts/download_data.sh
pip install -e .
python -m compileall src scripts tests
```

Planned programmatic usage after Stage 2 implementation:

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
```

## Project Structure

- `config/`: Default configuration for detectors, VLM judging, XAI methods, and evaluation.
- `src/`: Core Python package for detection, judging, explanation, selection, evaluation, feedback, and shared utilities.
- `scripts/`: Dataset download, baseline reproduction, selector training, ablations, and figure generation entry points.
- `tests/`: Pytest fixtures plus interface-level smoke tests for the scaffold.
- `notebooks/`: Research notebooks for exploratory analysis and result visualization.
- `data/`: Runtime data, COCO assets, and checkpoints created by `scripts/download_data.sh`.
- `results/`: Experiment outputs, figures, and ablation-specific result folders.

## Citation

```bibtex
@misc{aed_xai_2026,
  title        = {AED-XAI: Auto-Explainable Object Detection Pipeline},
  author       = {TODO},
  year         = {2026},
  howpublished = {TODO},
  note         = {Citation placeholder}
}
```
