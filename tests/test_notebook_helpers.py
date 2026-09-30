"""Tests for src/notebook_utils.py and src/v2_figures.py (headless, synthetic runs)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("sklearn")

from src import notebook_utils as nbu
from test_evidence_pipeline import _synthetic_run

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _synthetic_evidence(root: Path, name: str = "coco_dev_toy") -> Path:
    run_dir = root / name
    _synthetic_run(run_dir, num_images=60, seed=2)
    (run_dir / "manifests").mkdir(parents=True, exist_ok=True)
    (run_dir / "manifests" / "shard_000.json").write_text(
        json.dumps({"images": {"count": 70}, "git": {"dirty": False, "commit": "0123456789abc"}})
    )
    return run_dir


def test_progress_timing_and_estimate(tmp_path: Path) -> None:
    run_dir = _synthetic_evidence(tmp_path)
    progress = nbu.evidence_progress([run_dir.name, "missing_run"], root=tmp_path)
    row = progress.set_index("run").loc[run_dir.name]
    assert row["expected"] == 70 and row["done"] == 60 and row["status"] == "running/incomplete"
    assert progress.set_index("run").loc["missing_run", "status"] == "not started"
    timing = nbu.timing_summary(run_dir.name, root=tmp_path)
    assert timing["images"] == 60 and timing["explained_per_image"] == pytest.approx(4.0)
    estimate = nbu.estimate_gpu_hours(run_dir.name, target_images=1000, max_explained=2, root=tmp_path)
    assert estimate["explained_per_image"] == pytest.approx(2.0)


def test_apply_tau_keeps_comments(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "eval.yaml"
    config.write_text(
        "evaluation:\n  feedback:\n    escalation:\n      # PROVISIONAL\n      tau: 0.65\n"
        "      tau_by_detector:\n        yolox-s: 0.6546   # keep me\n        fasterrcnn_resnet50_fpn_v2: 0.6922\n\n"
        "# Experiment settings\nexperiments:\n  seed: 42\n"
    )
    tau = tmp_path / "tau.json"
    tau.write_text(json.dumps({"tau_by_detector": {"yolox-s": 0.61, "fasterrcnn_resnet50_fpn_v2": 0.7, "new-det": 0.5}}))
    monkeypatch.setattr(nbu, "PROJECT_ROOT", tmp_path)
    preview = nbu.apply_tau_to_eval_config("tau.json", "eval.yaml", dry_run=True)
    assert "+         yolox-s: 0.6100   # keep me" in preview and "0.6546" in config.read_text()
    nbu.apply_tau_to_eval_config("tau.json", "eval.yaml", dry_run=False)
    import yaml

    text = config.read_text()
    loaded = yaml.safe_load(text)["evaluation"]["feedback"]["escalation"]["tau_by_detector"]
    assert loaded == {"yolox-s": 0.61, "fasterrcnn_resnet50_fpn_v2": 0.7, "new-det": 0.5}
    assert "# PROVISIONAL" in text and "# keep me" in text and "# Experiment settings" in text


def test_suggest_methods_gate_summary_and_figures(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    run_dir = _synthetic_evidence(tmp_path, "toy_run")
    spec = importlib.util.spec_from_file_location("analyze_evidence", SCRIPTS / "analyze_evidence.py")
    analyze = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analyze)
    from src.evidence_io import detections_frame, images_for_ap

    out = tmp_path / "analysis"
    out.mkdir()
    cfg = {"cv_folds": 3, "bootstrap_samples": 20, "ap_bootstrap_samples": 3, "crc_alpha": [0.1], "acquisition_budgets": [0.0, 0.5, 1.0]}
    summary = analyze.analyze_detector(detections_frame([run_dir]), images_for_ap([run_dir]), "toy_run", cfg, out)
    (out / "summary.json").write_text(json.dumps([summary], default=float))

    lite = nbu.suggest_lite_methods(out, "toy_run")
    assert {"method", "auroc_gain_over_S2", "keep_for_lite"} <= set(lite.columns)
    gates = nbu.gate_summary(out)
    assert gates.loc[0, "run"] == "toy_run" and "G1 dAUROC" in gates

    from src import v2_figures as vf

    vf.apply_style()
    figures = [
        vf.fig_g1_forest(summary),
        vf.fig_g2_acquisition(out / "g2_acquisition_toy_run.csv"),
        vf.fig_abstention(out / "abstention_crc_toy_run.csv", 0.1),
    ]
    paths = vf.save_figure(figures[0], "g1", tmp_path / "figs")
    assert all(path.exists() for path in paths)
    assert (out / "g1_predictions_toy_run.csv").exists()


def test_spread_keeps_labels_apart() -> None:
    from src.v2_figures import _spread

    placed = _spread([0.30, 0.301, 0.20, 0.199], 0.05)
    ordered = sorted(placed)
    assert all(b - a >= 0.05 - 1e-9 for a, b in zip(ordered, ordered[1:]))
