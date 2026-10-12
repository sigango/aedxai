"""End-to-end smoke tests for the evidence runner, loader, and analysis scripts."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("sklearn")

from src.evaluator import AutoEvaluator
from src.evidence_io import detections_frame, images_for_ap
from src.gt_matching import AnnotationIndex
from src.utils import save_image
from src.xai_methods import get_explainer
from toy_models import ToyDetectorWrapper, toy_image

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _annotations(tmp_path: Path, names: list[str]) -> AnnotationIndex:
    path = tmp_path / "ann.json"
    path.write_text(
        json.dumps(
            {
                "images": [{"id": i + 1, "file_name": name, "width": 64, "height": 64} for i, name in enumerate(names)],
                "annotations": [
                    {"id": i + 1, "image_id": i + 1, "category_id": 1, "bbox": [16, 16, 32, 32], "iscrowd": 0}
                    for i in range(len(names))
                ],
                "categories": [{"id": 1, "name": "person"}, {"id": 3, "name": "car"}],
            }
        )
    )
    return AnnotationIndex.from_coco_json(path)


def test_process_image_writes_complete_record(tmp_path: Path) -> None:
    run_evidence = _load_script("run_evidence")
    image_path = tmp_path / "000001.jpg"
    save_image(toy_image(), str(image_path))
    explainers = {name: get_explainer(name, {}) for name in ("gcame", "gcame_raw", "bbox_prior", "random")}
    settings = {
        "conf_thresh": 0.25,
        "max_explained_per_image": 20,
        "gradient_features": True,
        "gt_iou_threshold": 0.5,
        "box_protocol": True,
        "shuffled_of": ["gcame"],
        "agreement_pairs": [("gcame", "gcame_raw")],
        "save_maps_for": ["gcame_raw"],
        "map_downsample": 2,
        "prenms": {"enabled": True, "conf_thresh": 0.01, "max_candidates": 100},
        "tta": {"enabled": True, "conf_thresh": 0.05, "views": ["hflip", "shrink", "shift", "dark"]},
    }
    record = run_evidence.process_image(
        image_path,
        detector=ToyDetectorWrapper(),
        evaluator=AutoEvaluator(),
        explainers=explainers,
        annotations=_annotations(tmp_path, ["000001.jpg"]),
        settings=settings,
        maps_dir=tmp_path / "maps",
    )
    labels = [det["match"]["label"] for det in record["detections"]]
    assert labels == ["tp", "fp"]
    first = record["detections"][0]
    assert set(first["methods"]) == {"gcame", "gcame_raw", "bbox_prior", "random", "shuffled_gcame"}
    assert "oa_box" in first["methods"]["gcame"] and "ebpg_gt" in first["methods"]["gcame"]
    assert first["gradient"]["grad_l2"] > 0 and "gcame|gcame_raw" in first["agreement"]
    # Cheap baseline tier for every detection, explained or not.
    assert first["prenms"]["count"] == 2.0 and first["prenms"]["other_class_count"] == 1.0
    assert set(first["tta"]) >= {"score_mean", "match_rate", "class_flip_rate", "score_hflip"}
    assert all("prenms" in det and "tta" in det for det in record["detections"])
    assert record["tta_time"] >= 0 and record["prenms_time"] >= 0
    saved = np.load(tmp_path / "maps" / "000001.npz")
    assert saved["det0__gcame_raw"].shape == (32, 32)
    json.dumps(record)  # must be JSON-serializable


def _synthetic_run(run_dir: Path, num_images: int = 80, seed: int = 0) -> None:
    """Fake evidence records where one explanation feature separates TP from FP."""
    rng = np.random.default_rng(seed)
    lines = []
    for index in range(num_images):
        detections, gts = [], []
        for det_id in range(4):
            is_tp = bool(rng.random() < 0.6)
            box = [10 * det_id, 10, 10 * det_id + 8, 18]
            if is_tp:
                gts.append({"bbox": box, "class_id": 0, "iscrowd": False})
            signal = (1.0 if is_tp else 0.0) + rng.normal(0, 0.5)
            fallback = bool(rng.random() < 0.2)
            method = lambda value: {  # noqa: E731
                "pg": 1.0, "ebpg": 0.5 + 0.1 * value, "oa": value, "insertion_auc": 0.5, "deletion_auc": 0.2,
                "sparsity": 0.8, "composite": 0.6, "oa_box": value, "composite_box": 0.6, "explain_time": 0.1,
                "eval_time": 0.2, "is_fallback": fallback, "insertion_curve": [0.1, 0.5, 0.9], "deletion_curve": [0.9, 0.4, 0.1],
                "map_features": {"mass_in_box": 0.5},
            }
            detections.append(
                {
                    "det_id": det_id, "bbox": box, "class_id": 0, "class_name": "person",
                    "confidence": float(np.clip(0.5 + rng.normal(0, 0.2), 0.01, 0.99)), "relative_size": "small",
                    "explained": True,
                    "match": {"label": "tp" if is_tp else "fp", "iou_gt": 1.0 if is_tp else 0.0, "gt_bbox": box if is_tp else None,
                              "fp_type": None if is_tp else "background", "best_iou_same_class": 0.0, "best_iou_any_class": 0.0},
                    "output": {"confidence": 0.5, "log_area": 4.0},
                    "gradient": {"grad_l2": float(rng.random()), "rg_head_l2": float(rng.random()), "grad_time": 0.01},
                    "prenms": {"count": float(rng.integers(1, 5)), "score_mean": float(rng.random())},
                    "tta": {"score_mean": float(rng.random()), "match_rate": float(rng.random())},
                    "methods": {"gcame": method(signal), "bbox_prior": method(rng.normal())},
                    "agreement": {},
                }
            )
        lines.append(json.dumps({"image": f"{index:06d}.jpg", "detector": "toy", "gt": gts, "detections": detections,
                                 "attributes": {}, "prenms_time": 0.04, "tta_time": 0.08}))
    (run_dir / "records").mkdir(parents=True)
    (run_dir / "records" / "shard_000.jsonl").write_text("\n".join(lines) + "\n")


def test_analysis_scripts_on_synthetic_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "toy_run"
    _synthetic_run(run_dir)
    frame = detections_frame([run_dir])
    assert len(frame) == 320 and "m__gcame__oa" in frame and "cf__gcame__del_first_drop" in frame

    assert "pre__count" in frame and "tta__match_rate" in frame and frame["cost__tta_s"].iloc[0] == pytest.approx(0.02)

    analyze = _load_script("analyze_evidence")
    sets = analyze.feature_sets(frame)
    assert {"S3_explanation", "P_controls_only", "S2_baseline", "S3_lite", "S3_map_only", "S3_curves_only"} <= set(sets)
    assert not any("_gt" in column for columns in sets.values() for column in columns)
    assert not any(column.startswith("cost__") for columns in sets.values() for column in columns)
    # Strong baseline = output + pre-NMS + all gradients + TTA; the lite baseline drops pre-NMS, TTA and rg_*.
    assert {"pre__count", "tta__match_rate", "grad__rg_head_l2", "grad__grad_l2"} <= set(sets["S2_baseline"])
    assert not {"pre__count", "tta__match_rate", "grad__rg_head_l2"} & set(sets["S2_output_gradient"])
    assert "m__gcame__is_fallback" in sets["S3_explanation"] and "m__gcame__is_fallback" not in sets["S3_no_fallback_flag"]
    assert not any(column.startswith(("cf__", "cfb__")) for column in sets["S3_map_only"])
    assert not any(column.startswith("mf__") for column in sets["S3_curves_only"])
    cfg = {"cv_folds": 3, "bootstrap_samples": 30, "ap_bootstrap_samples": 5, "crc_alpha": [0.1], "acquisition_budgets": [0.0, 0.5, 1.0],
           "claims": {"H1": {"pair": "S3_explanation - S2_baseline", "rule": "upper", "margin": 0.02},
                      "H4": {"pair": "S2_plus_vlm_box - S2_baseline", "rule": "lower", "margin": 0.0}}}
    out = tmp_path / "analysis"
    out.mkdir()
    summary = analyze.analyze_detector(frame, images_for_ap([run_dir]), "toy_run", cfg, out)
    assert summary["g1"]["gate"] == "S3_explanation - S2_baseline"
    # The toy explanation carries the TP signal, so "no gain" is rejected; the toy run has no VLM pass.
    assert [(c["claim"], c["verdict"]) for c in summary["claims"]] == [("H1", "NOT SUPPORTED"), ("H4", "NOT ESTIMABLE")]
    assert summary["g1"]["bootstrap"]["S3_explanation - S2_baseline"]["delta_auroc"] > 0.05
    assert summary["g1"]["bootstrap"]["S3_lite - S2_output_gradient"]["delta_auroc"] > 0.05
    assert (out / "fp_type_auroc_toy_run.csv").exists() and summary["g1"]["fp_type_auroc"][0]["fp_type"] == "background"
    assert "S2_baseline" in summary["g3"]["ap50_delta_vs_confidence"] and "ap50_delta_vs_S2_baseline" in summary["g3"]
    assert (out / "g2_acquisition_toy_run.csv").exists() and (out / "abstention_crc_toy_run.csv").exists()
    assert "ap50" in summary["g3"]

    validity = _load_script("analyze_metric_validity")
    table = validity.method_table(frame, samples=20, rng=np.random.default_rng(0))
    assert set(table["method"]) == {"gcame", "bbox_prior"}
    assert bool(table.set_index("method").loc["bbox_prior", "is_control"])
    fallbacks = validity.fallback_table(frame[frame["explained"]]).set_index("method")
    assert 0.1 < fallbacks.loc["gcame", "fallback_rate"] < 0.3
    assert {"composite_non_fallback", "bbox_prior_composite_fallback_rows", "tp_rate_fallback"} <= set(fallbacks.columns)

    audit_cfg = {
        "composite": {"reference": "bbox_prior", "metric": "composite", "rule": "noninferior", "margin": 0.01},
        "oa": {"reference": "bbox_prior", "metric": "oa", "rule": "noninferior", "margin": 0.01},
        "raw": {"pair": ["gcame_raw", "gradcam_raw"], "metric": "composite", "rule": "equivalent", "margin": 0.01},
    }
    audit = validity.audit_claims(frame[frame["explained"]], audit_cfg, samples=50, rng=np.random.default_rng(0))
    rows = audit.set_index("claim")
    # Equal composites (0.6 each); the toy gcame OA tracks the TP signal, the bbox prior's is noise around 0.
    assert rows.loc["composite", "comparison"] == "bbox_prior - gcame" and rows.loc["composite", "delta"] == pytest.approx(0.0)
    assert bool(rows.loc["composite", "holds"]) and not bool(rows.loc["oa", "holds"])
    assert "delta_non_fallback" in audit
    assert [(v["claim"], v["verdict"]) for v in validity.claim_verdicts(audit)] == [
        ("composite", "SUPPORTED"), ("oa", "NOT SUPPORTED"), ("raw", "NOT ESTIMABLE")]


def test_analyze_evidence_cli_writes_report(tmp_path: Path, monkeypatch) -> None:
    import sys

    import yaml

    run_dir = tmp_path / "toy_run"
    _synthetic_run(run_dir, num_images=60, seed=1)
    config = tmp_path / "evidence.yaml"
    config.write_text(yaml.safe_dump({"analysis": {
        "cv_folds": 3, "bootstrap_samples": 20, "ap_bootstrap_samples": 3, "crc_alpha": [0.1], "acquisition_budgets": [0.0, 1.0],
        "claims": {"H1_no_gain": {"pair": "S3_explanation - S2_baseline", "rule": "upper", "margin": 0.02}},
        "audit_claims": {"H3a": {"reference": "bbox_prior", "metric": "composite", "rule": "noninferior", "margin": 0.01}},
    }}))
    out = tmp_path / "analysis"
    analyze = _load_script("analyze_evidence")
    monkeypatch.setattr(sys, "argv", ["analyze_evidence.py", "--run-dirs", str(run_dir), "--output-dir", str(out),
                                      "--evidence-config", str(config)])
    analyze.main()
    report = (out / "report.md").read_text()
    assert "G1" in report and "toy_run" in report and "**H1_no_gain**" in report
    assert json.loads((out / "summary.json").read_text())[0]["detector"] == "toy_run"

    validity = _load_script("analyze_metric_validity")
    monkeypatch.setattr(sys, "argv", ["analyze_metric_validity.py", "--run-dirs", str(run_dir), "--output-dir",
                                      str(tmp_path / "validity"), "--bootstrap-samples", "10", "--evidence-config", str(config)])
    validity.main()
    assert "**H3a** (1 comparisons) -> SUPPORTED" in (tmp_path / "validity" / "report.md").read_text()
    assert (tmp_path / "validity" / "audit_claims_toy_run.csv").exists()
    assert (tmp_path / "validity" / "fallback_toy_run.csv").exists()


def test_evaluate_claims_rules() -> None:
    analyze = _load_script("analyze_evidence")
    bootstrap = {
        "S3 - S2": {"delta_auroc": 0.004, "ci_low": -0.002, "ci_high": 0.012},
        "V - S2": {"delta_auroc": 0.03, "ci_low": 0.025, "ci_high": 0.035},
    }
    claims = {
        "no_gain": {"pair": "S3 - S2", "rule": "upper", "margin": 0.02},
        "no_gain_violated": {"pair": "V - S2", "rule": "upper", "margin": 0.02},
        "gain": {"pair": "V - S2", "rule": "lower", "margin": 0.0},
        "gain_absent": {"pair": "S3 - S2", "rule": "lower", "margin": 0.0},
        "missing": {"pair": "X - S2", "rule": "upper", "margin": 0.02},
    }
    verdicts = {row["claim"]: row["verdict"] for row in analyze.evaluate_claims(bootstrap, claims)}
    assert verdicts == {
        "no_gain": "SUPPORTED",
        "no_gain_violated": "NOT SUPPORTED",
        "gain": "SUPPORTED",
        "gain_absent": "NOT SUPPORTED",
        "missing": "NOT ESTIMABLE",
    }
    with pytest.raises(ValueError):
        analyze.evaluate_claims(bootstrap, {"bad": {"pair": "S3 - S2", "rule": "two-sided", "margin": 0.0}})
