"""Gates G1-G3 for the explanation-as-evidence study (CPU only).

Reads one or more evidence runs (``scripts/run_evidence.py``, optionally joined
with ``scripts/run_vlm_verification.py``) and, per detector:

G1  Does explanation evidence predict TP vs FP beyond output and gradient
    uncertainty? Image-grouped K-fold out-of-fold predictions for nested feature
    sets, AUROC/AUPRC/FPR@95/ECE, and a paired image bootstrap for the AUROC gain
    of explanation features over the output+gradient baseline. Placebo sets use
    only model-free control maps, so a gain can be attributed to explanation
    content rather than to perturbation probing alone.
G2  Budget-aware evidence acquisition: when only a fraction of detections can
    get the expensive tier, does choosing them by predicted value of information
    beat uncertainty and random choice at the same compute?
G3  Does re-scoring detections with the evidence model raise AP (re-matched)?
Plus conformal risk control abstention (per-image accepted-FP rate <= alpha)
with per-subgroup TP coverage, and per-subgroup AUROC.

GT-derived columns (``*_gt``, ``iou_gt``, labels) are never used as features.

Example:
    python scripts/analyze_evidence.py \
        --run-dirs results/evidence/coco_test_yolox-s results/evidence/coco_test_fasterrcnn_resnet50_fpn_v2 \
        --output-dir results/analysis/evidence_coco_test
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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evidence_io import detections_frame, images_for_ap
from src.gt_matching import average_precision

logger = logging.getLogger("analyze_evidence")

CONTROLS = ("bbox_prior", "center_prior", "random")
EXCLUDED_SUBSTRINGS = ("_gt", "explain_time", "eval_time", "grad_time", "vlm__time", "grad_score")
METHOD_METRICS = (
    "pg",
    "ebpg",
    "oa",
    "insertion_auc",
    "deletion_auc",
    "sparsity",
    "composite",
    "oa_box",
    "insertion_auc_box",
    "deletion_auc_box",
    "composite_box",
    "is_fallback",
)


# ----------------------------------------------------------------------------- features


def _numeric(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    keep = []
    for column in columns:
        if column not in frame.columns or any(token in column for token in EXCLUDED_SUBSTRINGS):
            continue
        if pd.api.types.is_bool_dtype(frame[column]) or pd.api.types.is_numeric_dtype(frame[column]):
            keep.append(column)
    return sorted(set(keep))


def methods_in(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    """(real explanation methods, control/placebo maps) present in the frame."""
    names = sorted({column.split("__")[1] for column in frame.columns if column.startswith("m__")})
    controls = [name for name in names if name in CONTROLS or name.startswith("shuffled_")]
    return [name for name in names if name not in controls], controls


def method_columns(frame: pd.DataFrame, methods: list[str]) -> list[str]:
    columns: list[str] = []
    for method in methods:
        columns += [f"m__{method}__{metric}" for metric in METHOD_METRICS]
        columns += [c for c in frame.columns if c.startswith((f"mf__{method}__", f"cf__{method}__", f"cfb__{method}__"))]
    for column in frame.columns:
        if column.startswith("ag__"):
            first, second = column.split("__")[1:3]
            if first in methods and second in methods:
                columns.append(column)
    return _numeric(frame, columns)


def feature_sets(frame: pd.DataFrame) -> dict[str, list[str]]:
    """Nested feature sets; S2 is the baseline every explanation set must beat."""
    methods, controls = methods_in(frame)
    output = _numeric(frame, [c for c in frame.columns if c.startswith("out__")] + ["class_id"])
    gradient = _numeric(frame, [c for c in frame.columns if c.startswith("grad__")])
    vlm_box = _numeric(frame, [c for c in frame.columns if c.startswith("vlm__box_")])
    vlm_all = _numeric(frame, [c for c in frame.columns if c.startswith("vlm__")])
    s2 = output + gradient
    sets: dict[str, list[str]] = {
        "S0_confidence": ["out__confidence"],
        "S1_output": output,
        "S2_output_gradient": s2,
        "S3_explanation": s2 + method_columns(frame, methods),
    }
    placebo = [name for name in controls if not name.startswith("shuffled_")]
    if placebo:
        sets["P_controls_only"] = s2 + method_columns(frame, placebo)
    shuffled = [name for name in controls if name.startswith("shuffled_")]
    if shuffled:
        sets["P_shuffled_only"] = s2 + method_columns(frame, shuffled)
    for method in methods:
        sets[f"S3_single_{method}"] = s2 + method_columns(frame, [method])
    if vlm_box:
        sets["S2_plus_vlm_box"] = s2 + vlm_box
    if vlm_all:
        sets["S4_explanation_vlm"] = sets["S3_explanation"] + vlm_all
    return {name: sorted(set(columns)) for name, columns in sets.items() if columns}


# ----------------------------------------------------------------------------- models


def _make_model(kind: str, seed: int) -> Any:
    if kind == "gbm":
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=seed
        )
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))


def image_folds(groups: np.ndarray, folds: int, seed: int) -> np.ndarray:
    """Assign each row a fold id so all detections of an image share a fold."""
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    fold_of_image = dict(zip(rng.permutation(unique), np.arange(unique.size) % folds))
    return np.array([fold_of_image[group] for group in groups])


def oof_predict(frame: pd.DataFrame, columns: list[str], y: np.ndarray, fold_ids: np.ndarray, kind: str, seed: int) -> np.ndarray:
    """Out-of-fold P(TP)."""
    x = frame[columns].astype(float).to_numpy()
    prediction = np.zeros(len(frame), dtype=np.float64)
    for fold in np.unique(fold_ids):
        test = fold_ids == fold
        train = ~test
        if np.unique(y[train]).size < 2:
            prediction[test] = float(y[train].mean())
            continue
        model = _make_model(kind, seed)
        model.fit(x[train], y[train])
        prediction[test] = model.predict_proba(x[test])[:, 1]
    return prediction


def classification_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    if np.unique(y).size < 2:
        return {"n": int(y.size), "tp_rate": float(y.mean())}
    order = np.argsort(-p)
    tpr = np.cumsum(y[order]) / max(1, y.sum())
    fpr = np.cumsum(1 - y[order]) / max(1, (1 - y).sum())
    fpr_at_95 = float(fpr[np.searchsorted(tpr, 0.95)]) if tpr[-1] >= 0.95 else 1.0
    bins = np.clip((p * 10).astype(int), 0, 9)
    ece = float(sum(abs(p[bins == b].mean() - y[bins == b].mean()) * (bins == b).mean() for b in range(10) if (bins == b).any()))
    return {
        "n": int(y.size),
        "tp_rate": float(y.mean()),
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "fpr_at_95tpr": fpr_at_95,
        "brier": float(brier_score_loss(y, p)),
        "ece": ece,
    }


def paired_bootstrap_auroc(
    y: np.ndarray,
    predictions: dict[str, np.ndarray],
    groups: np.ndarray,
    comparisons: list[tuple[str, str]],
    samples: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    """Image-level bootstrap of AUROC(treatment) - AUROC(baseline)."""
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    rows_by_group = {group: np.flatnonzero(groups == group) for group in unique}
    deltas: dict[str, list[float]] = {f"{t} - {b}": [] for t, b in comparisons}
    for _ in range(samples):
        rows = np.concatenate([rows_by_group[g] for g in rng.choice(unique, size=unique.size, replace=True)])
        if np.unique(y[rows]).size < 2:
            continue
        cache: dict[str, float] = {}
        for treatment, baseline in comparisons:
            for name in (treatment, baseline):
                if name not in cache:
                    cache[name] = roc_auc_score(y[rows], predictions[name][rows])
            deltas[f"{treatment} - {baseline}"].append(cache[treatment] - cache[baseline])
    summary = {}
    for key, values in deltas.items():
        array = np.asarray(values)
        treatment, baseline = key.split(" - ")
        point = roc_auc_score(y, predictions[treatment]) - roc_auc_score(y, predictions[baseline])
        summary[key] = {
            "delta_auroc": float(point),
            "ci_low": float(np.percentile(array, 2.5)) if array.size else float("nan"),
            "ci_high": float(np.percentile(array, 97.5)) if array.size else float("nan"),
            "bootstrap_samples": int(array.size),
        }
    return summary


# ----------------------------------------------------------------------------- G2 acquisition


def acquisition_curves(
    frame: pd.DataFrame,
    y: np.ndarray,
    fold_ids: np.ndarray,
    p_cheap: np.ndarray,
    p_full: np.ndarray,
    cheap_columns: list[str],
    budgets: list[float],
    costs: dict[str, float],
    seed: int,
) -> pd.DataFrame:
    """AUROC when only a budget fraction of detections receives the expensive tier."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import roc_auc_score

    eps = 1e-6
    loss_cheap = -(y * np.log(np.clip(p_cheap, eps, 1)) + (1 - y) * np.log(np.clip(1 - p_cheap, eps, 1)))
    loss_full = -(y * np.log(np.clip(p_full, eps, 1)) + (1 - y) * np.log(np.clip(1 - p_full, eps, 1)))
    gain = loss_cheap - loss_full

    x = np.column_stack([frame[cheap_columns].astype(float).to_numpy(), p_cheap])
    voi = np.zeros(len(frame))
    for fold in np.unique(fold_ids):
        test = fold_ids == fold
        model = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, random_state=seed)
        model.fit(x[~test], gain[~test])
        voi[test] = model.predict(x[test])

    rng = np.random.default_rng(seed)
    policies = {
        "uncertainty": -np.abs(p_cheap - 0.5),
        "learned_voi": voi,
    }
    rows = []
    n = len(frame)
    for budget in budgets:
        k = int(round(budget * n))
        cost = costs["cheap"] + budget * costs["expensive"]
        for name, priority in policies.items():
            chosen = np.zeros(n, dtype=bool)
            if k > 0:
                chosen[np.argsort(-priority)[:k]] = True
            combined = np.where(chosen, p_full, p_cheap)
            rows.append({"budget": budget, "policy": name, "mean_cost_s": cost, "auroc": roc_auc_score(y, combined)})
        random_scores = []
        for _ in range(20):
            chosen = np.zeros(n, dtype=bool)
            if k > 0:
                chosen[rng.choice(n, size=k, replace=False)] = True
            random_scores.append(roc_auc_score(y, np.where(chosen, p_full, p_cheap)))
        rows.append(
            {
                "budget": budget,
                "policy": "random",
                "mean_cost_s": cost,
                "auroc": float(np.mean(random_scores)),
                "auroc_std": float(np.std(random_scores)),
            }
        )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- G3 AP


def ap_rescoring(
    images: list[dict[str, Any]],
    score_maps: dict[str, dict[tuple[str, int], float]],
    thresholds: tuple[float, ...] = (0.5, 0.75),
    bootstrap_samples: int = 200,
    seed: int = 0,
) -> dict[str, Any]:
    """AP with detections re-scored by each score map (unmapped detections keep confidence)."""

    def _rescored(subset: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
        mapping = score_maps.get(name, {})
        out = []
        for image in subset:
            dets = [
                {**det, "rescored": mapping.get((image["image"], det["det_id"]), det["score"])}
                for det in image["detections"]
            ]
            out.append({"gt": image["gt"], "detections": dets})
        return out

    names = ["confidence"] + [name for name in score_maps if name != "confidence"]
    results: dict[str, Any] = {}
    for threshold in thresholds:
        key = f"ap{int(threshold * 100)}"
        results[key] = {
            name: average_precision(_rescored(images, name), iou_threshold=threshold, score_key="rescored")["ap"]
            for name in names
        }
    rng = np.random.default_rng(seed)
    deltas: dict[str, list[float]] = {name: [] for name in names if name != "confidence"}
    for _ in range(bootstrap_samples):
        subset = [images[i] for i in rng.integers(0, len(images), size=len(images))]
        base = average_precision(_rescored(subset, "confidence"), iou_threshold=0.5, score_key="rescored")["ap"]
        for name in deltas:
            deltas[name].append(
                average_precision(_rescored(subset, name), iou_threshold=0.5, score_key="rescored")["ap"] - base
            )
    results["ap50_delta_vs_confidence"] = {
        name: {
            "delta": results["ap50"][name] - results["ap50"]["confidence"],
            "ci_low": float(np.percentile(values, 2.5)) if values else float("nan"),
            "ci_high": float(np.percentile(values, 97.5)) if values else float("nan"),
        }
        for name, values in deltas.items()
    }
    return results


# ----------------------------------------------------------------------------- abstention


def crc_threshold(losses_by_lambda: np.ndarray, lambdas: np.ndarray, alpha: float) -> float:
    """Conformal risk control: smallest lambda with (n/(n+1)) R_hat + 1/(n+1) <= alpha."""
    n = losses_by_lambda.shape[0]
    risk = losses_by_lambda.mean(axis=0)
    valid = np.flatnonzero((n / (n + 1.0)) * risk + 1.0 / (n + 1.0) <= alpha)
    return float(lambdas[valid[0]]) if valid.size else float("inf")


def abstention_analysis(
    frame: pd.DataFrame,
    scores: dict[str, np.ndarray],
    alphas: list[float],
    subgroup_columns: list[str],
    repeats: int = 20,
    seed: int = 0,
) -> pd.DataFrame:
    """Accept detections with score >= lambda; control per-image accepted-FP fraction."""
    lambdas = np.linspace(0.0, 1.0, 201)
    images = frame["image"].to_numpy()
    unique = np.unique(images)
    fp = (~frame["is_tp"].to_numpy()).astype(float)
    tp = frame["is_tp"].to_numpy()
    rows = []
    rng = np.random.default_rng(seed)
    index_by_image = {image: np.flatnonzero(images == image) for image in unique}
    for repeat in range(repeats):
        permuted = rng.permutation(unique)
        calib_images, test_images = permuted[: unique.size // 2], permuted[unique.size // 2 :]
        calib_rows = np.concatenate([index_by_image[i] for i in calib_images])
        test_rows = np.concatenate([index_by_image[i] for i in test_images])
        for name, score in scores.items():
            accepted = score[:, None] >= lambdas[None, :]
            per_detection_loss = accepted * fp[:, None]
            image_losses = np.stack(
                [per_detection_loss[index_by_image[i]].mean(axis=0) for i in calib_images], axis=0
            )
            for alpha in alphas:
                lam = crc_threshold(image_losses, lambdas, alpha)
                accept_test = score[test_rows] >= lam
                test_image_risk = np.mean(
                    [(score[index_by_image[i]] >= lam).astype(float).dot(fp[index_by_image[i]]) / len(index_by_image[i]) for i in test_images]
                )
                row = {
                    "repeat": repeat,
                    "score": name,
                    "alpha": alpha,
                    "lambda": lam,
                    "test_risk": float(test_image_risk),
                    "accepted_fraction": float(accept_test.mean()),
                    "tp_coverage": float(accept_test[tp[test_rows]].mean()) if tp[test_rows].any() else float("nan"),
                    "fp_rate_among_accepted": float(fp[test_rows][accept_test].mean()) if accept_test.any() else 0.0,
                }
                for column in subgroup_columns:
                    values = frame[column].to_numpy()[test_rows]
                    for value in pd.unique(values):
                        mask = (values == value) & tp[test_rows]
                        if mask.sum() >= 20:
                            row[f"tp_coverage[{column}={value}]"] = float(accept_test[mask].mean())
                rows.append(row)
    return pd.DataFrame(rows).groupby(["score", "alpha"]).mean(numeric_only=True).reset_index()


# ----------------------------------------------------------------------------- driver


def _class_frequency_tercile(frame: pd.DataFrame) -> pd.Series:
    counts = frame["class_id"].map(frame["class_id"].value_counts())
    ranks = counts.rank(method="dense", pct=True)
    return pd.cut(ranks, bins=[0, 1 / 3, 2 / 3, 1.0], labels=["rare", "mid", "frequent"], include_lowest=True).astype(str)


def analyze_detector(
    frame_all: pd.DataFrame,
    ap_images: list[dict[str, Any]],
    detector: str,
    cfg: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Run G1-G3 and abstention for one group (a run directory = detector x domain x split)."""
    seed = int(cfg.get("seed", 0))
    folds = int(cfg.get("cv_folds", 5))
    boot = int(cfg.get("bootstrap_samples", 2000))
    frame_all = frame_all[frame_all["label"] != "ignore"].reset_index(drop=True)
    frame = frame_all[frame_all["explained"]].reset_index(drop=True)
    frame["class_freq"] = _class_frequency_tercile(frame)
    y = frame["is_tp"].to_numpy().astype(int)
    groups = frame["image"].to_numpy()
    fold_ids = image_folds(groups, folds, seed)
    sets = feature_sets(frame)
    logger.info("[%s] %d explained detections (%.1f%% TP), %d images, sets: %s", detector, len(frame), 100 * y.mean(), np.unique(groups).size, list(sets))

    predictions: dict[str, np.ndarray] = {}
    metric_rows = []
    for name, columns in sets.items():
        for kind in ("gbm", "logreg") if name in {"S0_confidence", "S2_output_gradient", "S3_explanation", "S4_explanation_vlm"} else ("gbm",):
            p = oof_predict(frame, columns, y, fold_ids, kind, seed)
            key = name if kind == "gbm" else f"{name}[logreg]"
            predictions[key] = p
            metric_rows.append({"set": key, "num_features": len(columns), **classification_metrics(y, p)})
    metrics_table = pd.DataFrame(metric_rows).sort_values("auroc", ascending=False)
    metrics_table.to_csv(output_dir / f"g1_metrics_{detector}.csv", index=False)
    # Out-of-fold P(TP) per detection, for error analysis and qualitative figures.
    keep = ["image", "det_id", "bbox", "class_name", "confidence", "coco_size", "is_tp", "fp_type", "iou_gt"]
    prediction_table = frame[[column for column in keep if column in frame]].copy()
    for name in ("S0_confidence", "S2_output_gradient", "S3_explanation", "S4_explanation_vlm"):
        if name in predictions:
            prediction_table[f"p_{name}"] = predictions[name]
    prediction_table.to_csv(output_dir / f"g1_predictions_{detector}.csv", index=False)

    comparisons = [("S3_explanation", "S2_output_gradient"), ("S2_output_gradient", "S1_output"), ("S1_output", "S0_confidence")]
    for placebo in ("P_controls_only", "P_shuffled_only"):
        if placebo in predictions:
            comparisons += [(placebo, "S2_output_gradient"), ("S3_explanation", placebo)]
    if "S4_explanation_vlm" in predictions:
        comparisons.append(("S4_explanation_vlm", "S3_explanation"))
    if "S2_plus_vlm_box" in predictions:
        comparisons.append(("S2_plus_vlm_box", "S2_output_gradient"))
    bootstrap = paired_bootstrap_auroc(y, predictions, groups, comparisons, boot, seed)
    gate_key = "S3_explanation - S2_output_gradient"
    gate = bootstrap[gate_key]
    g1_pass = gate["delta_auroc"] >= float(cfg.get("gate_delta_auroc", 0.02)) and gate["ci_low"] > 0

    subgroup_rows = []
    from sklearn.metrics import roc_auc_score

    for column in ["coco_size", "class_freq"] + [c for c in frame.columns if c.startswith("attr__")]:
        for value, rows in frame.groupby(column).groups.items():
            rows = np.asarray(list(rows))
            if rows.size < 50 or np.unique(y[rows]).size < 2:
                continue
            subgroup_rows.append(
                {
                    "column": column,
                    "value": value,
                    "n": int(rows.size),
                    "tp_rate": float(y[rows].mean()),
                    **{
                        f"auroc_{name}": float(roc_auc_score(y[rows], predictions[name][rows]))
                        for name in ("S0_confidence", "S2_output_gradient", "S3_explanation")
                    },
                }
            )
    pd.DataFrame(subgroup_rows).to_csv(output_dir / f"subgroup_auroc_{detector}.csv", index=False)

    methods, _ = methods_in(frame)
    expensive_cost = float(
        sum(frame[f"m__{m}__explain_time"].mean() + frame[f"m__{m}__eval_time"].mean() for m in methods if f"m__{m}__explain_time" in frame)
    )
    costs = {
        "cheap": float(frame["grad__grad_time"].mean()) if "grad__grad_time" in frame else 0.0,
        "expensive": expensive_cost,
    }
    acquisition = acquisition_curves(
        frame,
        y,
        fold_ids,
        predictions["S2_output_gradient"],
        predictions["S3_explanation"],
        sets["S2_output_gradient"],
        list(cfg.get("acquisition_budgets", [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0])),
        costs,
        seed,
    )
    acquisition.to_csv(output_dir / f"g2_acquisition_{detector}.csv", index=False)
    per_method_cost = [
        {
            "method": m,
            "explain_time_s": float(frame[f"m__{m}__explain_time"].mean()),
            "eval_time_s": float(frame[f"m__{m}__eval_time"].mean()),
            "auroc_single": float(metrics_table.set_index("set").loc[f"S3_single_{m}", "auroc"]),
            "fallback_rate": float(frame[f"m__{m}__is_fallback"].mean()) if f"m__{m}__is_fallback" in frame else float("nan"),
        }
        for m in methods
        if f"S3_single_{m}" in metrics_table["set"].values
    ]
    pd.DataFrame(per_method_cost).to_csv(output_dir / f"g2_method_value_{detector}.csv", index=False)

    budgets_df = acquisition.pivot_table(index="budget", columns="policy", values="auroc")
    full = float(budgets_df.loc[1.0].max()) if 1.0 in budgets_df.index else float("nan")
    base = float(budgets_df.loc[0.0].max()) if 0.0 in budgets_df.index else float("nan")
    g2 = {"auroc_none": base, "auroc_all": full}
    if 0.5 in budgets_df.index and full > base:
        for policy in budgets_df.columns:
            g2[f"gain_retained_at_50pct[{policy}]"] = float((budgets_df.loc[0.5, policy] - base) / (full - base))

    # G3: re-score all detections of this detector (explained -> full model; others -> output model).
    y_all = frame_all["is_tp"].to_numpy().astype(int)
    folds_all = image_folds(frame_all["image"].to_numpy(), folds, seed)
    p_output_all = oof_predict(frame_all, sets["S1_output"], y_all, folds_all, "gbm", seed)
    keys_all = list(zip(frame_all["image"], frame_all["det_id"]))
    keys_explained = list(zip(frame["image"], frame["det_id"]))
    output_map = dict(zip(keys_all, p_output_all))
    score_maps = {"S1_output": output_map}
    for name in ("S2_output_gradient", "S3_explanation", "S4_explanation_vlm"):
        if name in predictions:
            score_maps[name] = {**output_map, **dict(zip(keys_explained, predictions[name]))}
    detector_images = [image for image in ap_images if image["run"] == detector]
    g3 = ap_rescoring(detector_images, score_maps, seed=seed, bootstrap_samples=int(cfg.get("ap_bootstrap_samples", 200)))

    abstention_scores = {"confidence": frame["confidence"].to_numpy()}
    for name in ("S2_output_gradient", "S3_explanation", "S4_explanation_vlm"):
        if name in predictions:
            abstention_scores[name] = predictions[name]
    subgroups = ["coco_size", "class_freq"] + [c for c in frame.columns if c.startswith("attr__")]
    abstention = abstention_analysis(frame, abstention_scores, list(cfg.get("crc_alpha", [0.05, 0.1])), subgroups, seed=seed)
    abstention.to_csv(output_dir / f"abstention_crc_{detector}.csv", index=False)

    return {
        "detector": detector,
        "num_explained": int(len(frame)),
        "num_images": int(np.unique(groups).size),
        "tp_rate": float(y.mean()),
        "g1": {
            "gate": gate_key,
            "threshold": float(cfg.get("gate_delta_auroc", 0.02)),
            "pass": bool(g1_pass),
            "bootstrap": bootstrap,
            "top_sets": metrics_table.head(8)[["set", "auroc", "auprc", "fpr_at_95tpr"]].to_dict(orient="records"),
        },
        "g2": {**g2, "costs_s_per_detection": costs},
        "g3": g3,
    }


def write_report(summary: list[dict[str, Any]], output_dir: Path) -> None:
    lines = ["# Explanation-as-evidence gates", ""]
    for item in summary:
        g1, g2, g3 = item["g1"], item["g2"], item["g3"]
        gate = g1["bootstrap"][g1["gate"]]
        lines += [
            f"## {item['detector']}",
            "",
            f"- Explained detections: {item['num_explained']} on {item['num_images']} images; TP rate {item['tp_rate']:.3f}",
            f"- **G1** {g1['gate']}: ΔAUROC {gate['delta_auroc']:+.4f} [95% CI {gate['ci_low']:+.4f}, {gate['ci_high']:+.4f}]"
            f" -> {'PASS' if g1['pass'] else 'FAIL'} (threshold {g1['threshold']})",
        ]
        for key, value in g1["bootstrap"].items():
            if key != g1["gate"]:
                lines.append(f"  - {key}: {value['delta_auroc']:+.4f} [{value['ci_low']:+.4f}, {value['ci_high']:+.4f}]")
        lines.append("- **G2** acquisition: " + ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in g2.items() if k != "costs_s_per_detection"))
        lines.append(f"  - cost per detection (s): {g2['costs_s_per_detection']}")
        lines.append("- **G3** AP50: " + ", ".join(f"{k}={v:.4f}" for k, v in g3["ap50"].items()))
        for name, value in g3["ap50_delta_vs_confidence"].items():
            lines.append(f"  - ΔAP50 {name}: {value['delta']:+.4f} [{value['ci_low']:+.4f}, {value['ci_high']:+.4f}]")
        lines.append("")
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dirs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--evidence-config", default="config/evidence_config.yaml")
    parser.add_argument("--bootstrap-samples", type=int, default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, format="[%(asctime)s %(name)s %(levelname)s] %(message)s")

    with Path(args.evidence_config).open("r", encoding="utf-8") as handle:
        cfg = dict((yaml.safe_load(handle) or {}).get("analysis", {}))
    if args.bootstrap_samples is not None:
        cfg["bootstrap_samples"] = args.bootstrap_samples
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = detections_frame(args.run_dirs)
    if frame.empty:
        raise SystemExit("No detections found in the given run directories.")
    ap_images = images_for_ap(args.run_dirs)
    # One analysis per run directory (detector x domain x split); AP never mixes datasets.
    summary = [
        analyze_detector(frame[frame["run"] == run].reset_index(drop=True), ap_images, run, cfg, output_dir)
        for run in sorted(frame["run"].unique())
    ]
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=float)
    write_report(summary, output_dir)
    print((output_dir / "report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
