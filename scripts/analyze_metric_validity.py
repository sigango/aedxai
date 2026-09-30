"""Audit of the annotation-free explanation metrics (CPU only).

Uses the same evidence runs as ``scripts/analyze_evidence.py`` and answers, per
run (detector x domain x split):

1. Controls vs methods: do model-free maps (bbox/center prior, random,
   pixel-shuffled) score like real explanations on the composite?
2. Object size: how do global OA and box-relative OA change from small to large
   objects (COCO area bins)? A trend that vanishes under the box-relative
   protocol points to metric geometry, not explanation quality.
3. Pseudo-GT validity: method ranking by EBPG against detector boxes vs COCO GT
   boxes (TP detections), and per-detection agreement.
4. Conditioning: gcame - gcame_raw and gradcam - gradcam_raw on every metric.
5. Headroom: best fixed method vs per-detection oracle under each objective.
6. Fallback rates and real explanation/evaluation times.

Bootstrap CIs resample images.

Example:
    python scripts/analyze_metric_validity.py \
        --run-dirs results/evidence/coco_test_yolox-s \
        --output-dir results/analysis/metric_validity_coco_test
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evidence_io import detections_frame

logger = logging.getLogger("analyze_metric_validity")

CONTROLS = ("bbox_prior", "center_prior", "random")
METRICS = ("composite", "ebpg", "oa", "sparsity", "pg", "oa_box", "composite_box", "ebpg_gt", "pg_gt")
OBJECTIVES = ("composite", "oa", "oa_box", "composite_box", "ebpg_gt")


def _markdown(table: pd.DataFrame) -> str:
    """Markdown table without requiring the optional ``tabulate`` package."""
    if table.empty:
        return "(empty)"
    header = "| " + " | ".join(str(column) for column in table.columns) + " |"
    divider = "|" + "|".join("---" for _ in table.columns) + "|"
    body = ["| " + " | ".join(str(value) for value in row) + " |" for row in table.itertuples(index=False)]
    return "\n".join([header, divider, *body])


def _names(frame: pd.DataFrame) -> list[str]:
    return sorted({column.split("__")[1] for column in frame.columns if column.startswith("m__")})


def _is_control(name: str) -> bool:
    return name in CONTROLS or name.startswith("shuffled_")


def _bootstrap_mean(values: np.ndarray, groups: np.ndarray, samples: int, rng: np.random.Generator) -> tuple[float, float]:
    mask = ~np.isnan(values)
    values, groups = values[mask], groups[mask]
    if values.size == 0:
        return float("nan"), float("nan")
    unique, inverse = np.unique(groups, return_inverse=True)
    sums = np.bincount(inverse, weights=values, minlength=unique.size)
    counts = np.bincount(inverse, minlength=unique.size)
    means = []
    for _ in range(samples):
        pick = rng.integers(0, unique.size, size=unique.size)
        means.append(sums[pick].sum() / max(1, counts[pick].sum()))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def method_table(frame: pd.DataFrame, samples: int, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    groups = frame["image"].to_numpy()
    for name in _names(frame):
        row: dict[str, Any] = {"method": name, "is_control": _is_control(name)}
        for metric in METRICS:
            column = f"m__{name}__{metric}"
            if column not in frame:
                continue
            values = frame[column].to_numpy(dtype=float)
            if metric in ("ebpg_gt", "pg_gt"):
                values = np.where(frame["is_tp"].to_numpy(), values, np.nan)
            low, high = _bootstrap_mean(values, groups, samples, rng)
            row[metric] = float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")
            row[f"{metric}_ci"] = f"[{low:.4f}, {high:.4f}]"
        for timing in ("explain_time", "eval_time", "is_fallback"):
            column = f"m__{name}__{timing}"
            if column in frame:
                row[timing] = float(frame[column].mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("composite", ascending=False)


def size_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in _names(frame):
        for size, part in frame.groupby("coco_size"):
            row = {"method": name, "coco_size": size, "n": int(len(part))}
            for metric in ("oa", "oa_box", "ebpg", "sparsity", "composite", "composite_box"):
                column = f"m__{name}__{metric}"
                if column in part:
                    row[metric] = float(part[column].mean())
            rows.append(row)
    table = pd.DataFrame(rows)
    order = {"small": 0, "medium": 1, "large": 2}
    return table.sort_values(["method", "coco_size"], key=lambda s: s.map(order) if s.name == "coco_size" else s)


def size_trend(table: pd.DataFrame) -> pd.DataFrame:
    """large - small difference of OA under each protocol, per method."""
    rows = []
    for name, part in table.groupby("method"):
        part = part.set_index("coco_size")
        if {"small", "large"} <= set(part.index):
            rows.append(
                {
                    "method": name,
                    "oa_large_minus_small": float(part.loc["large", "oa"] - part.loc["small", "oa"]),
                    "oa_box_large_minus_small": float(part.loc["large", "oa_box"] - part.loc["small", "oa_box"])
                    if "oa_box" in part
                    else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def pseudo_gt_agreement(frame: pd.DataFrame) -> dict[str, Any]:
    from scipy.stats import spearmanr

    tp = frame[frame["is_tp"]]
    real = [name for name in _names(frame) if not _is_control(name) and f"m__{name}__ebpg_gt" in frame]
    if len(real) < 2 or tp.empty:
        return {}
    det_means = [tp[f"m__{name}__ebpg"].mean() for name in real]
    gt_means = [tp[f"m__{name}__ebpg_gt"].mean() for name in real]
    per_detection = {
        name: float(np.corrcoef(tp[f"m__{name}__ebpg"], tp[f"m__{name}__ebpg_gt"])[0, 1]) for name in real
    }
    return {
        "methods": real,
        "ranking_spearman_det_vs_gt_box": float(spearmanr(det_means, gt_means).correlation),
        "ebpg_det_box": dict(zip(real, map(float, det_means))),
        "ebpg_gt_box": dict(zip(real, map(float, gt_means))),
        "per_detection_pearson": per_detection,
        "mean_iou_det_vs_gt": float(tp["iou_gt"].mean()),
    }


def conditioning_deltas(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for conditioned, raw in (("gcame", "gcame_raw"), ("gradcam", "gradcam_raw")):
        if f"m__{conditioned}__composite" not in frame or f"m__{raw}__composite" not in frame:
            continue
        row = {"pair": f"{conditioned} - {raw}"}
        for metric in METRICS:
            a, b = f"m__{conditioned}__{metric}", f"m__{raw}__{metric}"
            if a in frame and b in frame:
                row[metric] = float((frame[a] - frame[b]).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def headroom_table(frame: pd.DataFrame) -> pd.DataFrame:
    real = [name for name in _names(frame) if not _is_control(name)]
    rows = []
    for objective in OBJECTIVES:
        columns = [f"m__{name}__{objective}" for name in real if f"m__{name}__{objective}" in frame]
        if len(columns) < 2:
            continue
        part = frame if objective != "ebpg_gt" else frame[frame["is_tp"]]
        matrix = part[columns].to_numpy(dtype=float)
        valid = ~np.isnan(matrix).any(axis=1)
        matrix = matrix[valid]
        if matrix.size == 0:
            continue
        means = matrix.mean(axis=0)
        names = np.asarray([column.split("__")[1] for column in columns])
        winners = pd.Series(names[matrix.argmax(axis=1)]).value_counts(normalize=True)
        rows.append(
            {
                "objective": objective,
                "n": int(matrix.shape[0]),
                "best_fixed": columns[int(means.argmax())].split("__")[1],
                "best_fixed_mean": float(means.max()),
                "oracle_mean": float(matrix.max(axis=1).mean()),
                "headroom": float(matrix.max(axis=1).mean() - means.max()),
                "winner_share": json.dumps({k: round(float(v), 3) for k, v in winners.items()}),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dirs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    logging.basicConfig(level="INFO", format="[%(asctime)s %(name)s %(levelname)s] %(message)s")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = detections_frame(args.run_dirs, include_unexplained=False)
    frame = frame[frame["label"] != "ignore"]
    rng = np.random.default_rng(args.seed)
    report = ["# Metric validity audit", ""]
    summary: dict[str, Any] = {}
    for run, part in frame.groupby("run"):
        part = part.reset_index(drop=True)
        methods = method_table(part, args.bootstrap_samples, rng)
        sizes = size_table(part)
        trend = size_trend(sizes)
        conditioning = conditioning_deltas(part)
        headroom = headroom_table(part)
        agreement = pseudo_gt_agreement(part)
        for name, table in (
            ("methods", methods),
            ("by_size", sizes),
            ("size_trend", trend),
            ("conditioning", conditioning),
            ("headroom", headroom),
        ):
            table.to_csv(output_dir / f"{name}_{run}.csv", index=False)

        real = methods[~methods["is_control"]]
        controls = methods[methods["is_control"]]
        best_control = controls.sort_values("composite", ascending=False).head(1)
        beaten = (
            int((real["composite"] <= float(best_control["composite"].iloc[0])).sum()) if not best_control.empty else 0
        )
        summary[run] = {
            "num_detections": int(len(part)),
            "best_control": best_control[["method", "composite"]].to_dict(orient="records"),
            "real_methods_at_or_below_best_control": beaten,
            "num_real_methods": int(len(real)),
            "pseudo_gt_agreement": agreement,
            "headroom": headroom.to_dict(orient="records"),
            "size_trend": trend.to_dict(orient="records"),
        }
        report += [
            f"## {run} ({len(part)} explained detections)",
            "",
            "### Methods and controls (mean over detections; TP only for *_gt)",
            methods[[c for c in ["method", "is_control", "composite", "ebpg", "oa", "oa_box", "sparsity", "ebpg_gt", "explain_time", "is_fallback"] if c in methods]]
            .round(4)
            .pipe(_markdown),
            "",
            f"Real methods scoring at or below the best control on the composite: {beaten}/{len(real)}",
            "",
            "### OA by COCO size (large - small)",
            _markdown(trend.round(4)) if not trend.empty else "(no size bins)",
            "",
            "### Headroom per objective",
            _markdown(headroom.round(4)) if not headroom.empty else "(not enough methods)",
            "",
            "### Conditioning deltas",
            _markdown(conditioning.round(4)) if not conditioning.empty else "(raw variants not run)",
            "",
            f"### Detector-box vs GT-box EBPG: {json.dumps(agreement, default=float)[:1500]}",
            "",
        ]
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=float)
    (output_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
