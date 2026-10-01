"""Paper figures for the v2 evidence experiments (used by notebooks/08_v2_analysis_and_figures.ipynb).

Style rules applied here:
- Colours follow the entity, not the rank: each method keeps one colour in every figure.
- Real explanation methods and model-free controls use two encodings (colour
  vs grey with hatching), since that contrast is what the metric audit is about.
- Thin marks, hairline solid grids, one y-axis per panel, and a legend whenever
  there are two or more series.
- Light colours (aqua, yellow) are below 3:1 contrast on the surface, so every
  figure also carries direct value labels, and the notebook shows its table.
- Saliency overlays use a single-hue sequential ramp, never a rainbow.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

# Reference categorical palette, light mode, fixed slot order (validated with the
# dataviz validator for adjacent-pair CVD separation).
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK = {
    "primary": "#0b0b0b",
    "secondary": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "surface": "#fcfcfb",
}
CONTROL_COLOR = INK["muted"]
METHOD_COLORS = {
    "gcame": SLOTS[0],
    "gcame_raw": SLOTS[1],
    "lime": SLOTS[2],
    "dclose": SLOTS[3],
    "gradcam": SLOTS[4],
    "gradcam_raw": SLOTS[5],
}
METHOD_MARKERS = {"gcame": "o", "gcame_raw": "s", "lime": "^", "dclose": "D", "gradcam": "v", "gradcam_raw": "P"}
SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SET_LABELS = {
    "S0_confidence": "confidence only",
    "S1_output": "output features",
    "S1_metadetect": "output + pre-NMS candidates",
    "S1_tta": "output + TTA",
    "S2_output_gradient": "output + gradient (lite)",
    "S2_baseline": "strong baseline (output, pre-NMS, gradient, TTA)",
    "S3_explanation": "+ explanation evidence",
    "S3_lite": "lite baseline + explanation evidence",
    "S3_map_only": "+ CAM map structure only",
    "S3_curves_only": "+ perturbation curves only",
    "S3_no_fallback_flag": "+ explanation evidence (no fallback flag)",
    "S2_plus_vlm_all": "+ VLM box and salient-pixel checks",
    "S4_explanation_vlm": "+ explanation + VLM",
    "P_controls_only": "+ control maps only (placebo)",
    "P_shuffled_only": "+ shuffled maps only (placebo)",
    "S2_plus_vlm_box": "+ VLM box check only",
}


def is_control(name: str) -> bool:
    return name in {"bbox_prior", "center_prior", "random"} or name.startswith("shuffled_")


def apply_style() -> None:
    """Quiet, print-friendly matplotlib defaults."""
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.facecolor": INK["surface"],
            "axes.facecolor": INK["surface"],
            "savefig.facecolor": INK["surface"],
            "axes.edgecolor": INK["axis"],
            "axes.labelcolor": INK["secondary"],
            "axes.titlecolor": INK["primary"],
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": INK["grid"],
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",
            "xtick.color": INK["muted"],
            "ytick.color": INK["muted"],
            "xtick.labelcolor": INK["secondary"],
            "ytick.labelcolor": INK["secondary"],
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "text.color": INK["primary"],
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "legend.frameon": False,
            "legend.fontsize": 8,
            "lines.linewidth": 2.0,
            "lines.solid_capstyle": "round",
            "hatch.color": "#ffffff",
            "hatch.linewidth": 1.0,
        }
    )


def _parse_ci(value: Any) -> tuple[float, float]:
    if isinstance(value, str) and value.startswith("["):
        low, high = value.strip("[]").split(",")
        return float(low), float(high)
    return float("nan"), float("nan")


def _spread(values: Sequence[float], min_gap: float) -> list[float]:
    """Nudge label positions apart (keeping order) so direct labels never overlap."""
    order = np.argsort(values)
    placed = np.asarray(values, dtype=float)[order].copy()
    for index in range(1, placed.size):
        placed[index] = max(placed[index], placed[index - 1] + min_gap)
    shift = (placed - np.asarray(values, dtype=float)[order]).mean() / 2.0
    result = np.empty_like(placed)
    result[order] = placed - shift
    return result.tolist()


def save_figure(fig: Any, name: str, fig_dir: str | Path, dpi: int = 200) -> list[Path]:
    """Save as PNG and PDF; returns the written paths."""
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths = [fig_dir / f"{name}.png", fig_dir / f"{name}.pdf"]
    for path in paths:
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return paths


# ----------------------------------------------------------------------------- gate B


def fig_metric_audit(methods_csv: str | Path, metric: str = "composite", title: str | None = None) -> Any:
    """Horizontal bars: every method and control on one metric, with bootstrap CIs."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    table = pd.read_csv(methods_csv)
    table = table[table[metric].notna()].sort_values(metric)
    fig, ax = plt.subplots(figsize=(6.2, 0.32 * len(table) + 1.1))
    positions = np.arange(len(table))
    for position, (_, row) in zip(positions, table.iterrows()):
        control = is_control(str(row["method"]))
        ax.barh(
            position,
            row[metric],
            height=0.55,
            color=CONTROL_COLOR if control else METHOD_COLORS.get(str(row["method"]), SLOTS[0]),
            hatch="///" if control else None,
            linewidth=0,
        )
        low, high = _parse_ci(row.get(f"{metric}_ci"))
        if np.isfinite(low):
            ax.plot([low, high], [position, position], color=INK["secondary"], linewidth=1.0)
        ax.text(row[metric], position + 0.33, f"{row[metric]:.3f}", fontsize=7, color=INK["secondary"], va="bottom", ha="right")
    ax.set_yticks(positions, table["method"])
    ax.set_xlabel(metric)
    ax.grid(axis="y", visible=False)
    ax.set_title(title or f"{metric}: explanation methods vs model-free controls", loc="left", pad=22)
    ax.legend(
        handles=[
            Patch(color=SLOTS[0], label="explanation method (colour = method)"),
            Patch(facecolor=CONTROL_COLOR, hatch="///", label="model-free control"),
        ],
        loc="lower left",
        bbox_to_anchor=(0.0, 1.0),
        ncols=2,
        borderaxespad=0.2,
    )
    fig.tight_layout()
    return fig


def fig_oa_by_size(
    by_size_csv: str | Path,
    methods: Sequence[str] = ("gcame", "gcame_raw", "lime", "bbox_prior"),
    title: str | None = None,
) -> Any:
    """Two panels (global vs box-relative OA) over COCO size bins; one line per method."""
    import matplotlib.pyplot as plt

    table = pd.read_csv(by_size_csv)
    order = ["small", "medium", "large"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    panels = (("oa", "Global OA (10% of image per step)"), ("oa_box", "Box-relative OA (10% of 2x box per step)"))
    pending: list[tuple[Any, list[tuple[str, float]]]] = []
    for ax, (column, panel_title) in zip(axes, panels):
        if column not in table:
            ax.set_visible(False)
            continue
        end_labels: list[tuple[str, float]] = []
        for method in methods:
            part = table[table["method"] == method].set_index("coco_size").reindex(order)
            if part[column].isna().all():
                continue
            control = is_control(method)
            ax.plot(
                order,
                part[column],
                color=CONTROL_COLOR if control else METHOD_COLORS.get(method, SLOTS[0]),
                marker="s" if control else METHOD_MARKERS.get(method, "o"),
                markersize=6,
                markeredgecolor=INK["surface"],
                markeredgewidth=1.5,
                label=method,
            )
            end_labels.append((method, float(part[column].dropna().iloc[-1])))
        pending.append((ax, end_labels))
        ax.set_title(panel_title, loc="left", fontsize=9)
        ax.set_xlabel("COCO object size")
        ax.set_xlim(-0.2, len(order) - 1 + 0.9)
    low, high = axes[0].get_ylim()
    for ax, end_labels in pending:
        for (method, _), y in zip(end_labels, _spread([value for _, value in end_labels], 0.075 * (high - low))):
            ax.text(len(order) - 1 + 0.12, y, method, fontsize=7, color=INK["secondary"], va="center")
    axes[0].set_ylabel("OA (insertion - deletion AUC)")
    axes[0].legend(loc="upper right")
    fig.suptitle(title or "Does the size trend in OA survive a box-relative protocol?", x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------- G1-G3


def fig_g1_forest(summary_entry: Mapping[str, Any], gate: float | None = None) -> Any:
    """Paired-bootstrap AUROC differences with 95% CIs; the gate comparison is first."""
    import matplotlib.pyplot as plt

    bootstrap = summary_entry["g1"]["bootstrap"]
    gate_key = summary_entry["g1"]["gate"]
    keys = [gate_key] + [key for key in bootstrap if key != gate_key]
    gate = float(summary_entry["g1"].get("threshold", 0.02) if gate is None else gate)

    def _label(key: str) -> str:
        treatment, baseline = key.split(" - ")
        return f"{SET_LABELS.get(treatment, treatment)}\n  vs {SET_LABELS.get(baseline, baseline)}"

    fig, ax = plt.subplots(figsize=(6.4, 0.55 * len(keys) + 1.0))
    positions = np.arange(len(keys))[::-1]
    for position, key in zip(positions, keys):
        value = bootstrap[key]
        ax.plot([value["ci_low"], value["ci_high"]], [position, position], color=SLOTS[0], linewidth=2)
        ax.plot(value["delta_auroc"], position, "o", color=SLOTS[0], markersize=7, markeredgecolor=INK["surface"], markeredgewidth=1.5)
        ax.annotate(f"{value['delta_auroc']:+.3f}", (value["ci_high"], position), xytext=(5, 0), textcoords="offset points", fontsize=7, color=INK["secondary"], va="center")
    ax.axvline(0.0, color=INK["axis"], linewidth=1)
    ax.axvline(gate, color=INK["muted"], linewidth=1)
    ax.text(gate, 0.005, f" gate {gate:+.2f}", transform=ax.get_xaxis_transform(), fontsize=7, color=INK["muted"], ha="left", va="bottom")
    ax.set_yticks(positions, [_label(key) for key in keys], fontsize=7)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("ΔAUROC for TP vs FP (paired image bootstrap, 95% CI)")
    ax.set_title(f"G1 — {summary_entry['detector']}", loc="left")
    fig.tight_layout()
    return fig


def fig_g2_acquisition(acquisition_csv: str | Path, title: str | None = None) -> Any:
    """AUROC when only a fraction of detections gets the expensive evidence tier."""
    import matplotlib.pyplot as plt

    table = pd.read_csv(acquisition_csv)
    styles = {"learned_voi": (SLOTS[0], "o"), "uncertainty": (SLOTS[1], "s"), "random": (CONTROL_COLOR, "^")}
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    for policy, (color, marker) in styles.items():
        part = table[table["policy"] == policy].sort_values("budget")
        if part.empty:
            continue
        half = part[np.isclose(part["budget"], 0.5)]
        label = policy.replace("_", " ") + (f"  ({half['auroc'].iloc[0]:.3f} at 50%)" if not half.empty else "")
        ax.plot(part["budget"], part["auroc"], color=color, marker=marker, markersize=6, markeredgecolor=INK["surface"], markeredgewidth=1.5, label=label)
        if policy == "random" and "auroc_std" in part:
            ax.fill_between(part["budget"], part["auroc"] - part["auroc_std"], part["auroc"] + part["auroc_std"], color=color, alpha=0.1, linewidth=0)
    if np.isclose(table["budget"], 0.5).any():
        ax.axvline(0.5, color=INK["axis"], linewidth=1)
        ax.text(0.5, 0.01, " 50% budget (gate G2)", transform=ax.get_xaxis_transform(), fontsize=7, color=INK["muted"], va="bottom")
    ax.set_xlabel("fraction of detections given expensive evidence")
    ax.set_ylabel("AUROC (TP vs FP)")
    ax.set_title(title or "G2 — evidence acquisition under a budget", loc="left")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def fig_abstention(abstention_csv: str | Path, alpha: float, title: str | None = None) -> Any:
    """TP coverage at a conformal risk level, per score (higher = fewer good detections rejected)."""
    import matplotlib.pyplot as plt

    table = pd.read_csv(abstention_csv)
    table = table[np.isclose(table["alpha"], alpha)].sort_values("tp_coverage")
    fig, ax = plt.subplots(figsize=(5.6, 0.45 * len(table) + 1.1))
    positions = np.arange(len(table))
    ax.barh(positions, table["tp_coverage"], height=0.5, color=SLOTS[0], linewidth=0)
    for position, (_, row) in zip(positions, table.iterrows()):
        ax.text(row["tp_coverage"] + 0.01, position, f"{row['tp_coverage']:.3f}  (risk {row['test_risk']:.3f})", fontsize=7, color=INK["secondary"], va="center")
    ax.set_yticks(positions, [SET_LABELS.get(name, name) for name in table["score"]])
    ax.set_xlim(0, 1.15)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"TP coverage at accepted-FP risk ≤ {alpha}")
    ax.set_title(title or "Abstention with conformal risk control", loc="left")
    fig.tight_layout()
    return fig


def fig_subgroup_auroc(subgroup_csv: str | Path, column: str, title: str | None = None) -> Any:
    """Grouped bars: AUROC per subgroup value for confidence, output+gradient, and + explanation."""
    import matplotlib.pyplot as plt

    table = pd.read_csv(subgroup_csv)
    table = table[table["column"] == column]
    natural = {"small": 0, "medium": 1, "large": 2, "rare": 0, "mid": 1, "frequent": 2}
    table = table.assign(_order=table["value"].astype(str).map(natural).fillna(99)).sort_values(["_order", "value"])
    baseline = ("auroc_S2_baseline", "strong baseline") if "auroc_S2_baseline" in table else ("auroc_S2_output_gradient", "output + gradient")
    series = [("auroc_S0_confidence", "confidence only"), baseline, ("auroc_S3_explanation", "+ explanation evidence")]
    values = list(table["value"].astype(str))
    fig, ax = plt.subplots(figsize=(max(4.0, 1.3 * len(values) + 1.5), 3.0))
    width = 0.26
    for index, (key, label) in enumerate(series):
        if key not in table:
            continue
        x = np.arange(len(values)) + (index - 1) * (width + 0.02)
        ax.bar(x, table[key], width=width, color=SLOTS[index], label=label, linewidth=0)
        for xi, value in zip(x, table[key]):
            ax.text(xi, value + 0.005, f"{value:.2f}", fontsize=6, color=INK["secondary"], ha="center", va="bottom")
    ax.set_xticks(np.arange(len(values)), [f"{v}\n(n={n})" for v, n in zip(values, table["n"])])
    low = float(np.nanmin(table[[k for k, _ in series if k in table]].to_numpy()))
    ax.set_ylim(max(0.0, low - 0.1), 1.07)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("AUROC (TP vs FP)")
    ax.set_title(title or f"AUROC by {column.replace('attr__', '')}", loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.22), ncols=3)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------- qualitative


def load_saved_maps(npz_path: str | Path, det_id: int, image_shape: Sequence[int]) -> dict[str, np.ndarray]:
    """Saved (downsampled float16) maps of one detection, resized back to the image."""
    import cv2

    maps: dict[str, np.ndarray] = {}
    with np.load(npz_path) as saved:
        for key in saved.files:
            prefix, _, method = key.partition("__")
            if prefix == f"det{det_id}":
                small = saved[key].astype(np.float32)
                maps[method] = cv2.resize(small, (int(image_shape[1]), int(image_shape[0])), interpolation=cv2.INTER_LINEAR)
    return maps


def show_detection_maps(
    image: np.ndarray,
    maps: Mapping[str, np.ndarray],
    bbox: Sequence[float],
    title: str = "",
    gt_bbox: Sequence[float] | None = None,
) -> Any:
    """Image with the detection box, then one overlay per explanation map."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle

    colors = [(*_hex_to_rgb(hex_color), alpha) for hex_color, alpha in zip(SEQUENTIAL_BLUE, np.linspace(0.0, 0.85, len(SEQUENTIAL_BLUE)))]
    cmap = LinearSegmentedColormap.from_list("aedxai_blue_alpha", colors)
    names = list(maps)
    fig, axes = plt.subplots(1, len(names) + 1, figsize=(2.3 * (len(names) + 1), 2.5))
    axes = np.atleast_1d(axes)
    for ax in axes:
        ax.imshow(image)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        x1, y1, x2, y2 = bbox
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=SLOTS[1], linewidth=1.5))
        if gt_bbox is not None:
            gx1, gy1, gx2, gy2 = gt_bbox
            ax.add_patch(Rectangle((gx1, gy1), gx2 - gx1, gy2 - gy1, fill=False, edgecolor=INK["surface"], linewidth=1.0))
    axes[0].set_title("detection (orange) / GT (white)", fontsize=7, loc="left")
    for ax, name in zip(axes[1:], names):
        ax.imshow(maps[name], cmap=cmap, vmin=0.0, vmax=1.0)
        ax.set_title(name, fontsize=8, loc="left")
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=9)
    fig.tight_layout()
    return fig


def _hex_to_rgb(hex_color: str) -> tuple[float, float, float]:
    value = hex_color.lstrip("#")
    return tuple(int(value[index : index + 2], 16) / 255.0 for index in (0, 2, 4))  # type: ignore[return-value]


def load_summary(analysis_dir: str | Path) -> list[dict[str, Any]]:
    with (Path(analysis_dir) / "summary.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


__all__ = [
    "INK",
    "METHOD_COLORS",
    "SLOTS",
    "apply_style",
    "fig_abstention",
    "fig_g1_forest",
    "fig_g2_acquisition",
    "fig_metric_audit",
    "fig_oa_by_size",
    "fig_subgroup_auroc",
    "load_saved_maps",
    "load_summary",
    "save_figure",
    "show_detection_maps",
]
