"""E3: does a verifier that reads raw saliency maps add TP/FP signal? (PLAN_V3_EVALUATION.md)

The pre-registered H1 tested 314 hand-crafted explanation features. E3 checks that
the negative result does not come from that feature design. A small CNN reads a
square, box-centred crop (side = ``context`` x the longer box side) of the image
and of the saliency maps saved by ``run_evidence.py``, with a box-mask channel,
and is trained with the same image-grouped folds as ``analyze_evidence.py``.

Variants (input channels, each plus the box mask):
    R    RGB crop
    RM   RGB crop + saliency maps (default: gcame, gcame_raw, lime, dclose)
    M    saliency maps only
Each CNN score is stacked with the strong baseline (out-of-fold P(TP) of
``S2_baseline`` from ``g1_predictions_*.csv``) by a logistic regression fitted on
the same folds. Primary, pre-registered contrast: ``S2+RM - S2+R`` (do the maps
add information beyond the image crop and the strong baseline?). E3 is SUPPORTED
when its paired image-bootstrap 95% CI upper bound is below ``--margin`` (0.02).

GT is only the TP/FP endpoint; the CNN never sees GT boxes.

Example:
    python scripts/analyze_map_verifier.py --run-dir results/evidence/coco_test_yolox-s \
        --predictions results/analysis/evidence_coco_test/g1_predictions_coco_test_yolox-s.csv \
        --images-dir data/coco/val2017 --output-dir results/analysis/e3_map_verifier_yolox-s
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analyze_evidence import classification_metrics, evaluate_claims, image_folds, paired_bootstrap_auroc  # noqa: E402
from src.utils import load_image  # noqa: E402

logger = logging.getLogger("analyze_map_verifier")

VARIANTS = {"R": ("rgb",), "RM": ("rgb", "maps"), "M": ("maps",)}
FILL_VALUE = 114  # neutral grey outside the image, as in the perturbation explainers


# ----------------------------------------------------------------------------- data


def _square_window(bbox: list[float], context: float) -> tuple[int, int, int]:
    """(x0, y0, side) of a square window centred on the box."""
    x1, y1, x2, y2 = bbox
    side = max(8, int(round(context * max(x2 - x1, y2 - y1))))
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    return int(round(cx - side / 2.0)), int(round(cy - side / 2.0)), side


def _crop(array: np.ndarray, x0: int, y0: int, side: int, fill: float) -> np.ndarray:
    """Crop ``side`` x ``side`` at (x0, y0) from an HxW[xC] array, padding with ``fill``."""
    height, width = array.shape[:2]
    out = np.full((side, side) + array.shape[2:], fill, dtype=array.dtype)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(width, x0 + side), min(height, y0 + side)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = array[sy0:sy1, sx0:sx1]
    return out


def _resize(array: np.ndarray, size: int) -> np.ndarray:
    import cv2

    interpolation = cv2.INTER_AREA if array.shape[0] > size else cv2.INTER_LINEAR
    return cv2.resize(array, (size, size), interpolation=interpolation)


def _full_resolution(saved: np.ndarray, height: int, width: int) -> np.ndarray:
    """Undo ``map_downsample`` (every k-th pixel) by repetition."""
    step_y = int(np.ceil(height / saved.shape[0]))
    step_x = int(np.ceil(width / saved.shape[1]))
    return np.repeat(np.repeat(saved.astype(np.float32), step_y, axis=0), step_x, axis=1)[:height, :width]


def build_dataset(
    predictions: pd.DataFrame,
    run_dir: Path,
    images_dir: Path,
    methods: list[str],
    size: int,
    context: float,
) -> dict[str, np.ndarray]:
    """Crops for every detection in ``predictions`` (one row per explained detection)."""
    n = len(predictions)
    rgb = np.zeros((n, 3, size, size), dtype=np.uint8)
    maps = np.zeros((n, len(methods), size, size), dtype=np.float16)
    mask = np.zeros((n, 1, size, size), dtype=np.uint8)
    missing = np.zeros((n, len(methods)), dtype=bool)
    for image_name, rows in predictions.groupby("image", sort=False):
        image = load_image(str(images_dir / image_name))
        height, width = image.shape[:2]
        npz_path = run_dir / "maps" / f"{Path(image_name).stem}.npz"
        saved = dict(np.load(npz_path)) if npz_path.exists() else {}
        for index, row in rows.iterrows():
            bbox = [float(value) for value in json.loads(row["bbox"])]
            x0, y0, side = _square_window(bbox, context)
            rgb[index] = _resize(_crop(image, x0, y0, side, FILL_VALUE), size).transpose(2, 0, 1)
            box = np.zeros((height, width), dtype=np.uint8)
            box[int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])] = 1
            mask[index, 0] = _resize(_crop(box, x0, y0, side, 0), size)
            for channel, method in enumerate(methods):
                key = f"det{int(row['det_id'])}__{method}"
                if key not in saved:
                    missing[index, channel] = True
                    continue
                full = _full_resolution(saved[key], height, width)
                maps[index, channel] = _resize(_crop(full, x0, y0, side, 0.0), size).astype(np.float16)
    return {"rgb": rgb, "maps": maps, "mask": mask, "missing": missing}


# ----------------------------------------------------------------------------- model


def _inputs(data: dict[str, np.ndarray], parts: tuple[str, ...]) -> np.ndarray:
    blocks = []
    if "rgb" in parts:
        blocks.append(data["rgb"].astype(np.float32) / 255.0)
    if "maps" in parts:
        blocks.append(data["maps"].astype(np.float32))
    blocks.append(data["mask"].astype(np.float32))
    return np.concatenate(blocks, axis=1)


def _make_cnn(in_channels: int):
    import torch.nn as nn

    layers: list[Any] = []
    channels = in_channels
    for width in (32, 64, 128, 128):
        layers += [nn.Conv2d(channels, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(inplace=True), nn.MaxPool2d(2)]
        channels = width
    return nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(channels, 1))


def oof_cnn_logits(
    x: np.ndarray, y: np.ndarray, fold_ids: np.ndarray, epochs: int, batch_size: int, lr: float, seed: int, device: str
) -> np.ndarray:
    """Out-of-fold CNN logits; channels standardised with training-fold statistics."""
    import torch

    logits = np.zeros(len(y), dtype=np.float64)
    for fold in np.unique(fold_ids):
        test, train = fold_ids == fold, fold_ids != fold
        torch.manual_seed(seed + int(fold))
        mean = x[train].mean(axis=(0, 2, 3), keepdims=True)
        std = x[train].std(axis=(0, 2, 3), keepdims=True) + 1e-6
        x_train = torch.from_numpy(((x[train] - mean) / std).astype(np.float32))
        y_train = torch.from_numpy(y[train].astype(np.float32))
        x_test = torch.from_numpy(((x[test] - mean) / std).astype(np.float32))
        model = _make_cnn(x.shape[1]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        generator = torch.Generator().manual_seed(seed + int(fold))
        for _ in range(epochs):
            model.train()
            order = torch.randperm(len(y_train), generator=generator)
            for start in range(0, len(order), batch_size):
                batch = order[start : start + batch_size]
                if len(batch) < 2:  # BatchNorm needs at least two samples
                    continue
                inputs = x_train[batch]
                flip = torch.rand(len(batch), generator=generator) < 0.5
                inputs[flip] = inputs[flip].flip(-1)
                optimizer.zero_grad()
                loss = loss_fn(model(inputs.to(device)).squeeze(1), y_train[batch].to(device))
                loss.backward()
                optimizer.step()
        model.eval()
        with torch.no_grad():
            outputs = [model(x_test[start : start + 512].to(device)).squeeze(1).cpu() for start in range(0, len(x_test), 512)]
        logits[test] = torch.cat(outputs).numpy() if outputs else np.zeros(0)
    return logits


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def oof_stack(features: np.ndarray, y: np.ndarray, fold_ids: np.ndarray) -> np.ndarray:
    """Out-of-fold P(TP) of a logistic regression on [logit P_S2, CNN logit]."""
    from sklearn.linear_model import LogisticRegression

    prediction = np.zeros(len(y), dtype=np.float64)
    for fold in np.unique(fold_ids):
        test, train = fold_ids == fold, fold_ids != fold
        model = LogisticRegression(max_iter=1000).fit(features[train], y[train])
        prediction[test] = model.predict_proba(features[test])[:, 1]
    return prediction


# ----------------------------------------------------------------------------- driver


def _device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True, help="Evidence run with saved maps (results/evidence/<run>).")
    parser.add_argument("--predictions", required=True, help="g1_predictions_<run>.csv from analyze_evidence.py.")
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--methods", default="gcame,gcame_raw,lime,dclose", help="Saved maps used as channels.")
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--context", type=float, default=2.0)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--margin", type=float, default=0.02)
    parser.add_argument("--max-images", type=int, default=0, help="Development cap (0 = all).")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    logging.basicConfig(level="INFO", format="[%(asctime)s %(name)s %(levelname)s] %(message)s")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(args.predictions)
    if args.max_images:
        keep = predictions["image"].drop_duplicates().iloc[: args.max_images]
        predictions = predictions[predictions["image"].isin(keep)]
    predictions = predictions.reset_index(drop=True)
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    run = Path(args.run_dir).name

    cache = output_dir / f"dataset_{run}_s{args.size}_c{args.context:g}.npz"
    if cache.exists():
        data = dict(np.load(cache))
    else:
        t_start = time.time()
        data = build_dataset(predictions, Path(args.run_dir), Path(args.images_dir), methods, args.size, args.context)
        np.savez_compressed(cache, **data)
        logger.info("Built %d crops in %.0f s -> %s", len(predictions), time.time() - t_start, cache)
    if data["missing"].any():
        logger.warning("Missing maps for %d detection x method pairs (left as zeros).", int(data["missing"].sum()))

    y = predictions["is_tp"].astype(bool).to_numpy().astype(int)
    groups = predictions["image"].to_numpy()
    fold_ids = image_folds(groups, args.folds, args.seed)
    device = _device(args.device)
    scores: dict[str, np.ndarray] = {"S2_baseline": predictions["p_S2_baseline"].to_numpy(dtype=float)}
    base_logit = _logit(scores["S2_baseline"])
    for name, parts in VARIANTS.items():
        t_start = time.time()
        cnn = oof_cnn_logits(_inputs(data, parts), y, fold_ids, args.epochs, args.batch_size, args.lr, args.seed, device)
        scores[name] = 1.0 / (1.0 + np.exp(-cnn))
        scores[f"S2+{name}"] = oof_stack(np.column_stack([base_logit, cnn]), y, fold_ids)
        logger.info("Variant %s done in %.0f s on %s", name, time.time() - t_start, device)

    comparisons = [("S2+RM", "S2+R"), ("S2+R", "S2_baseline"), ("S2+M", "S2_baseline"), ("S2+RM", "S2_baseline"), ("RM", "R")]
    bootstrap = paired_bootstrap_auroc(y, scores, groups, comparisons, args.bootstrap_samples, args.seed)
    claims = evaluate_claims(bootstrap, {"E3_maps_add_no_signal": {"pair": "S2+RM - S2+R", "rule": "upper", "margin": args.margin}})
    metrics = pd.DataFrame([{"score": name, **classification_metrics(y, values)} for name, values in scores.items()])
    out = predictions[["image", "det_id", "is_tp"]].copy()
    for name, values in scores.items():
        out[f"p_{name}"] = values
    out.to_csv(output_dir / f"e3_predictions_{run}.csv", index=False)
    metrics.to_csv(output_dir / f"e3_metrics_{run}.csv", index=False)
    summary = {
        "run": run,
        "num_detections": int(len(y)),
        "num_images": int(np.unique(groups).size),
        "tp_rate": float(y.mean()),
        "methods": methods,
        "settings": {key: getattr(args, key) for key in ("size", "context", "epochs", "batch_size", "lr", "folds", "seed")},
        "device": device,
        "missing_maps": int(data["missing"].sum()),
        "bootstrap": bootstrap,
        "claims": claims,
        "metrics": metrics.to_dict(orient="records"),
    }
    with (output_dir / f"summary_{run}.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=float)
    lines = [f"# E3 raw-map verifier: {run}", "", f"- {len(y)} explained detections on {summary['num_images']} images; TP rate {y.mean():.3f}"]
    for claim in claims:
        estimate = f"{claim['delta_auroc']:+.4f} [{claim['ci_low']:+.4f}, {claim['ci_high']:+.4f}]" if "delta_auroc" in claim else "not estimable"
        lines.append(f"- **{claim['claim']}** ({claim['pair']}; CI high < {claim['margin']}): {estimate} -> {claim['verdict']}")
    for key, value in bootstrap.items():
        lines.append(f"  - {key}: {value['delta_auroc']:+.4f} [{value['ci_low']:+.4f}, {value['ci_high']:+.4f}]")
    lines += ["", "| score | auroc | auprc | fpr_at_95tpr |", "|---|---|---|---|"]
    lines += [f"| {row['score']} | {row.get('auroc', float('nan')):.4f} | {row.get('auprc', float('nan')):.4f} | {row.get('fpr_at_95tpr', float('nan')):.4f} |" for row in summary["metrics"]]
    (output_dir / f"report_{run}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
