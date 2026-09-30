"""Helpers for the v2 notebooks (06-08).

The notebooks never define experiment commands themselves: they call
``cluster/task.sh`` (one job) or ``cluster/submit.sh`` (a stage), exactly as the
cluster does, and use the functions below to inspect progress and results.
Only pandas/numpy are imported at module level.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_ROOT = PROJECT_ROOT / "results" / "evidence"
ANALYSIS_ROOT = PROJECT_ROOT / "results" / "analysis"
DETECTORS = {"yolox": "yolox-s", "frcnn": "fasterrcnn_resnet50_fpn_v2"}


# ----------------------------------------------------------------------------- running


def run_command(
    argv: Sequence[str],
    *,
    env: Mapping[str, str] | None = None,
    dry_run: bool = False,
    cwd: Path = PROJECT_ROOT,
) -> int:
    """Run a command from the project root, streaming its output into the notebook."""
    printable = " ".join(str(part) for part in argv)
    print(f"$ {printable}")
    if dry_run:
        print("  (dry run: not executed)")
        return 0
    full_env = os.environ.copy()
    full_env.setdefault("PYTHONUNBUFFERED", "1")
    full_env.update(dict(env or {}))
    process = subprocess.Popen(
        [str(part) for part in argv],
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=full_env,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
    code = process.wait()
    print(f"[exit {code}] {printable[:120]}")
    return code


def ensure_env_file(env_file: str | Path = "cluster/cluster.env") -> Path:
    """Create cluster/cluster.env from the example if it does not exist yet."""
    path = PROJECT_ROOT / env_file
    if not path.exists():
        example = PROJECT_ROOT / "cluster" / "cluster.env.example"
        path.write_text(example.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"Created {path.relative_to(PROJECT_ROOT)} from the example - edit PYTHON / RUN_PREFIX / paths.")
    return path


def load_cluster_env(env_file: str | Path = "cluster/cluster.env") -> dict[str, str]:
    """Resolved variables of the env file (bash does the expansion)."""
    path = ensure_env_file(env_file)
    script = f'set -a; source "{path}" >/dev/null 2>&1; env -0'
    output = subprocess.run(["bash", "-c", script], cwd=PROJECT_ROOT, capture_output=True, check=True).stdout
    values = dict(item.split("=", 1) for item in output.decode().split("\0") if "=" in item)
    keys = set(re.findall(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=", path.read_text(), flags=re.M))
    return {key: values.get(key, "") for key in sorted(keys)}


def run_task(
    task: str,
    shard: int = 0,
    num_shards: int = 1,
    *,
    env_file: str = "cluster/cluster.env",
    extra_env: Mapping[str, str] | None = None,
    dry_run: bool = False,
) -> int:
    """Run one task (one shard) exactly as the cluster would."""
    ensure_env_file(env_file)
    env = {"AEDXAI_ENV": env_file, **dict(extra_env or {})}
    return run_command(["bash", "cluster/task.sh", task, str(shard), str(num_shards)], env=env, dry_run=dry_run)


def submit(
    stages: Iterable[str],
    *,
    backend: str | None = None,
    env_file: str = "cluster/cluster.env",
    extra_env: Mapping[str, str] | None = None,
    dry_run: bool = False,
) -> int:
    """Launch stages with cluster/submit.sh (SLURM: returns after submission; local: blocks)."""
    ensure_env_file(env_file)
    env = {"AEDXAI_ENV": env_file, **dict(extra_env or {})}
    if backend:
        env["BACKEND"] = backend
    return run_command(["bash", "cluster/submit.sh", *list(stages)], env=env, dry_run=dry_run)


def run_name(split: str, detector_key: str, domain: str = "coco") -> str:
    """Run directory name used by cluster/task.sh, e.g. coco_test_yolox-s."""
    return f"{domain}_{split}_{DETECTORS.get(detector_key, detector_key)}"


# ----------------------------------------------------------------------------- progress / timing


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def evidence_progress(run_names: Iterable[str], root: Path = EVIDENCE_ROOT) -> pd.DataFrame:
    """Per-shard progress: expected images (from shard manifests), done, errors, detections, time."""
    rows = []
    for name in run_names:
        run_dir = Path(root) / name
        manifests = sorted((run_dir / "manifests").glob("shard_*.json"))
        if not manifests:
            rows.append({"run": name, "shard": None, "status": "not started"})
            continue
        for manifest_path in manifests:
            with manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            shard = manifest_path.stem
            records_path = run_dir / "records" / f"{shard}.jsonl"
            records = _read_jsonl(records_path) if records_path.exists() else []
            ok = {r["image"]: r for r in records if "error" not in r}
            errors = {r["image"] for r in records if "error" in r} - set(ok)
            expected = int(manifest["images"]["count"])
            times = [float(r.get("image_time", 0.0)) for r in ok.values()]
            rows.append(
                {
                    "run": name,
                    "shard": shard,
                    "expected": expected,
                    "done": len(ok),
                    "failed_pending_retry": len(errors),
                    "pct": round(100.0 * len(ok) / max(1, expected), 1),
                    "detections": sum(len(r.get("detections", [])) for r in ok.values()),
                    "explained": sum(sum(d.get("explained", False) for d in r.get("detections", [])) for r in ok.values()),
                    "mean_image_s": round(float(np.mean(times)), 2) if times else None,
                    "git_dirty": manifest.get("git", {}).get("dirty"),
                    "commit": (manifest.get("git", {}).get("commit") or "")[:10],
                    "status": "complete" if len(ok) >= expected else "running/incomplete",
                }
            )
    return pd.DataFrame(rows)


def timing_summary(run: str, root: Path = EVIDENCE_ROOT) -> dict[str, Any]:
    """Measured cost: seconds per image, per explained detection, and per method."""
    run_dir = Path(root) / run
    records = [r for path in sorted((run_dir / "records").glob("shard_*.jsonl")) for r in _read_jsonl(path) if "error" not in r]
    if not records:
        return {"run": run, "images": 0}
    image_times = np.array([float(r.get("image_time", 0.0)) for r in records])
    detections = np.array([len(r.get("detections", [])) for r in records])
    explained = np.array([sum(d.get("explained", False) for d in r.get("detections", [])) for r in records])
    per_method: dict[str, list[float]] = {}
    for record in records:
        for det in record.get("detections", []):
            for method, values in dict(det.get("methods", {})).items():
                per_method.setdefault(method, []).append(
                    float(values.get("explain_time", 0.0)) + float(values.get("eval_time", 0.0))
                )
    return {
        "run": run,
        "images": int(len(records)),
        "s_per_image": float(image_times.mean()),
        "detections_per_image": float(detections.mean()),
        "explained_per_image": float(explained.mean()),
        "s_per_explained_detection": float(image_times.sum() / max(1, explained.sum())),
        "detection_count_distribution": np.percentile(detections, [50, 90, 99]).tolist(),
        "s_per_method_per_detection": {method: round(float(np.mean(values)), 3) for method, values in sorted(per_method.items())},
    }


def estimate_gpu_hours(
    run: str,
    target_images: int,
    max_explained: int | None = None,
    cost_scale: float = 1.0,
    root: Path = EVIDENCE_ROOT,
) -> dict[str, float]:
    """Project GPU-hours for another split from a measured run.

    ``cost_scale`` rescales the per-detection cost, e.g. lite/full evidence
    forwards (about 970 / 1200 = 0.8).
    """
    summary = timing_summary(run, root)
    if not summary.get("images"):
        raise ValueError(f"No finished images in {run}")
    run_dir = Path(root) / run
    counts = [
        len(r.get("detections", []))
        for path in sorted((run_dir / "records").glob("shard_*.jsonl"))
        for r in _read_jsonl(path)
        if "error" not in r
    ]
    explained_per_image = float(np.mean([min(c, max_explained) if max_explained else c for c in counts]))
    seconds = target_images * explained_per_image * summary["s_per_explained_detection"] * cost_scale
    return {
        "target_images": target_images,
        "explained_per_image": round(explained_per_image, 2),
        "s_per_explained_detection": round(summary["s_per_explained_detection"] * cost_scale, 2),
        "gpu_hours": round(seconds / 3600.0, 1),
    }


# ----------------------------------------------------------------------------- decisions


def suggest_lite_methods(analysis_dir: str | Path, run: str, min_gain: float = 0.005) -> pd.DataFrame:
    """Rank methods by AUROC gain over S2 when added alone, and by gain per second of cost."""
    analysis_dir = Path(analysis_dir)
    metrics = pd.read_csv(analysis_dir / f"g1_metrics_{run}.csv").set_index("set")
    value = pd.read_csv(analysis_dir / f"g2_method_value_{run}.csv")
    base = float(metrics.loc["S2_output_gradient", "auroc"])
    value["auroc_gain_over_S2"] = value["auroc_single"] - base
    value["cost_s"] = value["explain_time_s"] + value["eval_time_s"]
    value["gain_per_second"] = value["auroc_gain_over_S2"] / value["cost_s"].clip(lower=1e-6)
    value["keep_for_lite"] = value["auroc_gain_over_S2"] >= min_gain
    return value.sort_values("auroc_gain_over_S2", ascending=False).reset_index(drop=True)


def apply_tau_to_eval_config(
    tau_json: str | Path = "results/analysis/tau_calibration.json",
    eval_config: str | Path = "config/eval_config.yaml",
    dry_run: bool = True,
) -> str:
    """Write calibrated tau_by_detector values into eval_config.yaml, keeping its comments.

    Returns a before/after listing of the changed lines; writes only when dry_run=False.
    """
    tau_path, config_path = PROJECT_ROOT / tau_json, PROJECT_ROOT / eval_config
    with tau_path.open("r", encoding="utf-8") as handle:
        calibrated = dict(json.load(handle)["tau_by_detector"])
    lines = config_path.read_text(encoding="utf-8").splitlines(keepends=True)
    try:
        start = next(index for index, line in enumerate(lines) if line.strip() == "tau_by_detector:")
    except StopIteration as exc:
        raise ValueError(f"No 'tau_by_detector:' block in {config_path}") from exc
    indent = len(lines[start]) - len(lines[start].lstrip())
    changes, seen = [], set()
    last_entry = start
    index = start + 1
    while index < len(lines):
        line = lines[index]
        if line.strip() and not line.lstrip().startswith("#") and len(line) - len(line.lstrip()) <= indent:
            break
        match = re.match(r"^(\s+)([A-Za-z0-9_.\-]+):\s*([0-9.]+)(.*)$", line.rstrip("\n"))
        if match and match.group(2) in calibrated:
            new_line = f"{match.group(1)}{match.group(2)}: {float(calibrated[match.group(2)]):.4f}{match.group(4)}\n"
            changes.append(f"- {line.rstrip()}\n+ {new_line.rstrip()}")
            lines[index] = new_line
            seen.add(match.group(2))
        if match:
            last_entry = index
        index += 1
    missing = set(calibrated) - seen
    if missing:
        insert = [f"{' ' * (indent + 2)}{name}: {float(calibrated[name]):.4f}\n" for name in sorted(missing)]
        lines[last_entry + 1 : last_entry + 1] = insert
        changes += [f"+ {line.rstrip()}" for line in insert]
    report = "\n".join(changes) if changes else "(no changes)"
    if not dry_run:
        config_path.write_text("".join(lines), encoding="utf-8")
        report += f"\nWrote {config_path.relative_to(PROJECT_ROOT)}. Update the PROVISIONAL comment above it and commit."
    return report


def config_is_committed(paths: Sequence[str] = ("config/evidence_config_lite.yaml", "config/eval_config.yaml", "manifests")) -> bool:
    """True when the frozen inputs have no uncommitted changes (checked before the test split)."""
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", *paths], cwd=PROJECT_ROOT, capture_output=True, text=True
    ).stdout.strip()
    if status:
        print("Uncommitted changes in frozen inputs:\n" + status)
    return not status


def gate_summary(analysis_dir: str | Path) -> pd.DataFrame:
    """One row per run with the G1/G2/G3 verdict numbers from analyze_evidence's summary.json."""
    with (Path(analysis_dir) / "summary.json").open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    rows = []
    for item in summary:
        gate = item["g1"]["bootstrap"][item["g1"]["gate"]]
        g3 = item["g3"].get("ap50_delta_vs_confidence", {}).get("S3_explanation", {})
        row = {
            "run": item["detector"],
            "explained": item["num_explained"],
            "tp_rate": round(item["tp_rate"], 3),
            "G1 dAUROC": round(gate["delta_auroc"], 4),
            "G1 CI": f"[{gate['ci_low']:+.4f}, {gate['ci_high']:+.4f}]",
            "G1 pass": item["g1"]["pass"],
            "G3 dAP50": round(g3.get("delta", float("nan")), 4),
            "G3 CI": f"[{g3.get('ci_low', float('nan')):+.4f}, {g3.get('ci_high', float('nan')):+.4f}]",
        }
        for key, value in item["g2"].items():
            if key.startswith("gain_retained_at_50pct"):
                row[key.replace("gain_retained_at_50pct", "G2 kept@50%")] = round(value, 3)
        rows.append(row)
    return pd.DataFrame(rows)


def python_executable() -> str:
    """The interpreter running this notebook (used for direct script calls)."""
    return sys.executable


__all__ = [
    "ANALYSIS_ROOT",
    "DETECTORS",
    "EVIDENCE_ROOT",
    "PROJECT_ROOT",
    "apply_tau_to_eval_config",
    "config_is_committed",
    "ensure_env_file",
    "estimate_gpu_hours",
    "evidence_progress",
    "gate_summary",
    "load_cluster_env",
    "python_executable",
    "run_command",
    "run_name",
    "run_task",
    "submit",
    "suggest_lite_methods",
    "timing_summary",
]
