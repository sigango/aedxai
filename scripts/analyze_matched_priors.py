"""E1 matched-prior audit (CPU only; PLAN_V3_EVALUATION.md).

Joins each frozen test run (real methods and default controls) with its E1 run
(``bbox_prior_sNNN`` controls: model-free Gaussians on the detection box with
sigma = NNN/100 x half the box side) on (image, det_id), and asks whether any real
explanation method beats a model-free bbox prior of matched sparsity.

Replication check: boxes and classes must agree, and ``bbox_prior_s100`` must
reproduce the test run's ``bbox_prior`` scores.

E1a (pre-registered): for every real method m, the prior k*(m) whose mean Gini
sparsity is closest to m's on the same detections is not worse than m on the
composite by more than ``noninferiority_margin`` (image-bootstrap CI low of
composite(prior_k*) - composite(m) > -margin).
E1b (reported): per-detection matching (the prior closest in sparsity on each
detection) gives control-calibrated scores, metric(m) - metric(matched prior),
for the composite, EBPG, OA and box-relative OA.

Example:
    python scripts/analyze_matched_priors.py \
        --run-pairs results/evidence/coco_test_yolox-s:results/evidence/e1priors_test_yolox-s \
        --config config/evidence_config_e1_priors.yaml --output-dir results/analysis/e1_matched_priors
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
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analyze_metric_validity import _bootstrap_mean, _is_control, _markdown, _names  # noqa: E402
from src.evidence_io import detections_frame  # noqa: E402

logger = logging.getLogger("analyze_matched_priors")

METRICS = ("composite", "ebpg", "oa", "oa_box", "sparsity")
PRIOR_PREFIX = "bbox_prior_s"


def prior_scale(name: str) -> float:
    return int(name[len(PRIOR_PREFIX):]) / 100.0


def join_runs(test_frame: pd.DataFrame, prior_frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Inner join on (image, det_id); drop rows whose box or class differ between the runs."""
    prior_columns = ["image", "det_id", "bbox", "class_id"] + [c for c in prior_frame.columns if c.startswith(f"m__{PRIOR_PREFIX}")]
    joined = test_frame.merge(prior_frame[prior_columns], on=["image", "det_id"], how="inner", suffixes=("", "__e1"))
    same = (joined["bbox"] == joined["bbox__e1"]) & (joined["class_id"] == joined["class_id__e1"])
    check: dict[str, Any] = {
        "test_rows": int(len(test_frame)),
        "e1_rows": int(len(prior_frame)),
        "joined_rows": int(len(joined)),
        "box_or_class_mismatch": int((~same).sum()),
    }
    joined = joined[same].drop(columns=["bbox__e1", "class_id__e1"]).reset_index(drop=True)
    first, second = "m__bbox_prior__composite", f"m__{PRIOR_PREFIX}100__composite"
    if first in joined and second in joined:
        diff = (joined[first] - joined[second]).abs()
        check["bbox_prior_vs_s100_max_abs_diff"] = float(diff.max()) if len(diff) else float("nan")
        check["bbox_prior_vs_s100_mean_abs_diff"] = float(diff.mean()) if len(diff) else float("nan")
    return joined, check


def prior_curve(frame: pd.DataFrame, priors: list[str]) -> pd.DataFrame:
    rows = []
    for name in priors:
        row: dict[str, Any] = {"prior": name, "sigma_scale": prior_scale(name)}
        for metric in METRICS:
            column = f"m__{name}__{metric}"
            if column in frame:
                row[metric] = float(frame[column].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def matched_by_mean_sparsity(
    frame: pd.DataFrame, methods: list[str], priors: list[str], margin: float, samples: int, rng: np.random.Generator
) -> pd.DataFrame:
    """E1a: one sparsity-matched prior per method, paired deltas with image-bootstrap CIs."""
    groups = frame["image"].to_numpy()
    rows = []
    for method in methods:
        sparsity = frame[f"m__{method}__sparsity"].to_numpy(dtype=float)
        valid = ~np.isnan(sparsity)
        gaps = {prior: abs(float(np.nanmean(frame[f"m__{prior}__sparsity"].to_numpy(dtype=float)[valid])) - float(np.nanmean(sparsity))) for prior in priors}
        best = min(gaps, key=gaps.get)
        row: dict[str, Any] = {"method": method, "matched_prior": best, "sparsity_gap": gaps[best]}
        for metric in METRICS:
            first, second = f"m__{best}__{metric}", f"m__{method}__{metric}"
            if first not in frame or second not in frame:
                continue
            diff = (frame[first] - frame[second]).to_numpy(dtype=float)
            row[f"{metric}_prior_minus_method"] = float(np.nanmean(diff))
            if metric == "composite":
                low, high = _bootstrap_mean(diff, groups, samples, rng)
                row.update(ci_low=low, ci_high=high, holds=bool(low > -margin), prior_superior=bool(low > 0))
        rows.append(row)
    return pd.DataFrame(rows)


def calibrated_per_detection(
    frame: pd.DataFrame, methods: list[str], priors: list[str], samples: int, rng: np.random.Generator
) -> pd.DataFrame:
    """E1b: metric(m) - metric(prior closest in sparsity on the same detection)."""
    groups = frame["image"].to_numpy()
    prior_sparsity = np.stack([frame[f"m__{prior}__sparsity"].to_numpy(dtype=float) for prior in priors], axis=1)
    rows = []
    for method in methods:
        own = frame[f"m__{method}__sparsity"].to_numpy(dtype=float)
        distance = np.abs(prior_sparsity - own[:, None])
        nearest = np.where(np.isnan(distance), np.inf, distance).argmin(axis=1)
        row: dict[str, Any] = {"method": method}
        for metric in ("composite", "ebpg", "oa", "oa_box"):
            column = f"m__{method}__{metric}"
            if column not in frame:
                continue
            prior_values = np.stack([frame[f"m__{prior}__{metric}"].to_numpy(dtype=float) for prior in priors], axis=1)
            matched = prior_values[np.arange(len(frame)), nearest]
            diff = frame[column].to_numpy(dtype=float) - matched
            low, high = _bootstrap_mean(diff, groups, samples, rng)
            row[f"calibrated_{metric}"] = float(np.nanmean(diff))
            row[f"calibrated_{metric}_ci"] = f"[{low:+.4f}, {high:+.4f}]"
        row["matched_prior_share"] = json.dumps(
            {priors[index]: round(float(share), 3) for index, share in enumerate(np.bincount(nearest, minlength=len(priors)) / len(frame)) if share > 0}
        )
        rows.append(row)
    return pd.DataFrame(rows)


def analyze_pair(test_dir: Path, prior_dir: Path, cfg: dict[str, Any], output_dir: Path, rng: np.random.Generator) -> dict[str, Any]:
    test_frame = detections_frame([test_dir], include_unexplained=False)
    test_frame = test_frame[test_frame["label"] != "ignore"].reset_index(drop=True)
    prior_frame = detections_frame([prior_dir], include_unexplained=False)
    frame, check = join_runs(test_frame, prior_frame)
    run = test_dir.name
    priors = sorted((name for name in _names(frame) if name.startswith(PRIOR_PREFIX)), key=prior_scale)
    methods = [name for name in _names(frame) if not _is_control(name)]
    if not priors or not methods:
        raise SystemExit(f"{run}: need real methods and {PRIOR_PREFIX}NNN priors after the join (got {methods}, {priors}).")
    samples = int(cfg.get("bootstrap_samples", 1000))
    margin = float(cfg.get("noninferiority_margin", 0.01))
    curve = prior_curve(frame, priors)
    matched = matched_by_mean_sparsity(frame, methods, priors, margin, samples, rng)
    calibrated = calibrated_per_detection(frame, methods, priors, samples, rng)
    for name, table in (("prior_curve", curve), ("matched", matched), ("calibrated", calibrated)):
        table.to_csv(output_dir / f"{name}_{run}.csv", index=False)
    verdict = "SUPPORTED" if bool(matched["holds"].all()) else "NOT SUPPORTED"
    method_means = {m: float(frame[f"m__{m}__composite"].mean()) for m in methods}
    best_method = max(method_means, key=method_means.get)
    best_prior = curve.sort_values("composite", ascending=False).iloc[0]
    return {
        "run": run,
        "num_detections": int(len(frame)),
        "replication": check,
        "E1a_verdict": verdict,
        "margin": margin,
        "best_method": {"method": best_method, "composite": method_means[best_method]},
        "best_prior": {"prior": str(best_prior["prior"]), "composite": float(best_prior["composite"])},
        "matched": matched.to_dict(orient="records"),
        "calibrated": calibrated.to_dict(orient="records"),
        "prior_curve": curve.to_dict(orient="records"),
    }


def write_report(summary: list[dict[str, Any]], output_dir: Path) -> None:
    lines = ["# E1 matched-prior audit", ""]
    for item in summary:
        check = item["replication"]
        matched = pd.DataFrame(item["matched"])
        lines += [
            f"## {item['run']} ({item['num_detections']} explained detections)",
            "",
            f"- Replication: {check['joined_rows']} joined of {check['test_rows']} test rows; "
            f"{check['box_or_class_mismatch']} box/class mismatches; "
            f"|bbox_prior - bbox_prior_s100| max {check.get('bbox_prior_vs_s100_max_abs_diff', float('nan')):.2e}",
            f"- **E1a** (sparsity-matched prior not worse than any method on the composite by > {item['margin']}): {item['E1a_verdict']}",
            f"- Best real method {item['best_method']['method']} ({item['best_method']['composite']:.4f}); "
            f"best prior {item['best_prior']['prior']} ({item['best_prior']['composite']:.4f})",
            "",
            "### Sparsity-matched prior minus method (E1a)",
            _markdown(matched.round(4)),
            "",
            "### Control-calibrated scores: method minus per-detection matched prior (E1b)",
            _markdown(pd.DataFrame(item["calibrated"]).round(4)),
            "",
            "### Prior width curve",
            _markdown(pd.DataFrame(item["prior_curve"]).round(4)),
            "",
        ]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-pairs", nargs="+", required=True, help="TEST_RUN_DIR:E1_RUN_DIR pairs, one per detector.")
    parser.add_argument("--config", default="config/evidence_config_e1_priors.yaml")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    logging.basicConfig(level="INFO", format="[%(asctime)s %(name)s %(levelname)s] %(message)s")
    config_path = Path(args.config)
    if not config_path.is_absolute() and not config_path.exists():
        config_path = PROJECT_ROOT / config_path
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = dict((yaml.safe_load(handle) or {}).get("analysis") or {})
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(cfg.get("seed", 0)))
    summary = []
    for pair in args.run_pairs:
        test_dir, prior_dir = (Path(part) for part in pair.split(":", 1))
        summary.append(analyze_pair(test_dir, prior_dir, cfg, output_dir, rng))
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=float)
    write_report(summary, output_dir)
    print((output_dir / "report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
