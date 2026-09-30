"""Calibrate the per-detector escalation tau on a held-out calibration split.

tau = percentile-p of the default method's per-detection composite. Use a run on
the manifest's ``calib`` split (never the test split), then paste the printed
block into ``config/eval_config.yaml: evaluation.feedback.escalation.tau_by_detector``.

Sources:
    --run-dirs      evidence runs (uses m__<method>__composite; detector from records)
    --progress-csv  legacy xai_selector_training_progress_<detector>.csv files
                    (composite recomputed exactly as AutoEvaluator.composite_score)

Example:
    python scripts/calibrate_tau.py --run-dirs results/evidence/coco_calib_yolox-s \
        results/evidence/coco_calib_fasterrcnn_resnet50_fpn_v2 --percentile 30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluator import AutoEvaluator
from src.evidence_io import detections_frame


def _from_progress(path: Path, method: str, evaluator: AutoEvaluator) -> tuple[str, np.ndarray]:
    frame = pd.read_csv(path)
    detector = path.stem.replace("xai_selector_training_progress_", "")
    scores = np.array(
        [
            evaluator.composite_score({"ebpg": row[f"ebpg_{method}"], "oa": row[f"oa_{method}"], "sparsity": row[f"sparsity_{method}"]})
            for _, row in frame.iterrows()
        ]
    )
    return detector, scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dirs", nargs="*", default=[])
    parser.add_argument("--progress-csv", nargs="*", default=[])
    parser.add_argument("--method", default="gcame", help="Default method whose composite sets tau.")
    parser.add_argument("--percentile", type=float, default=30.0)
    parser.add_argument("--eval-config", default="config/eval_config.yaml")
    parser.add_argument("--output", default=None, help="Optional JSON file for the calibrated values.")
    args = parser.parse_args()

    evaluator = AutoEvaluator(args.eval_config)
    scores_by_detector: dict[str, np.ndarray] = {}
    sources: dict[str, str] = {}
    if args.run_dirs:
        frame = detections_frame(args.run_dirs, include_unexplained=False)
        column = f"m__{args.method}__composite"
        if column not in frame:
            raise SystemExit(f"{column} not found; was '{args.method}' in evidence.methods?")
        for detector, part in frame.groupby("detector"):
            scores_by_detector[str(detector)] = part[column].dropna().to_numpy()
            sources[str(detector)] = ",".join(sorted(part["run"].unique()))
    for raw_path in args.progress_csv:
        detector, scores = _from_progress(Path(raw_path), args.method, evaluator)
        scores_by_detector[detector] = scores
        sources[detector] = str(raw_path)

    if not scores_by_detector:
        raise SystemExit("Pass --run-dirs and/or --progress-csv.")
    taus = {detector: float(np.percentile(scores, args.percentile)) for detector, scores in scores_by_detector.items()}
    report = {
        "method": args.method,
        "percentile": args.percentile,
        "composite_weights": evaluator.composite_weights,
        "tau_by_detector": {k: round(v, 4) for k, v in taus.items()},
        "n": {k: int(v.size) for k, v in scores_by_detector.items()},
        "sources": sources,
    }
    print(json.dumps(report, indent=2))
    print("\n# paste under evaluation.feedback.escalation in config/eval_config.yaml")
    print(yaml.safe_dump({"tau_by_detector": report["tau_by_detector"]}, sort_keys=True))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
