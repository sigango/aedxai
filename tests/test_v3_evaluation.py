"""Tests for the v3 evaluation additions: scaled bbox priors and the matched-prior audit (E1),
and the raw-map verifier (E3)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("sklearn")

from src.detector import Detection
from src.evaluator import AutoEvaluator
from src.utils import save_image
from src.xai_methods import base_method_name, get_explainer, is_control_method, scaled_prior_name

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _detection() -> Detection:
    return Detection(bbox=[20, 10, 60, 40], class_id=0, class_name="person", confidence=0.9, area=1200, relative_size="medium", detection_id=1)


def test_scaled_bbox_priors_sharpen_with_smaller_sigma() -> None:
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    default = get_explainer("bbox_prior", {}).explain(None, image, _detection())
    scaled = get_explainer("bbox_prior_s100", {}).explain(None, image, _detection())
    assert np.array_equal(default.map, scaled.map) and scaled.method_name == "bbox_prior_s100"
    evaluator = AutoEvaluator()
    gini = [evaluator.sparsity_gini(get_explainer(scaled_prior_name(k), {}).explain(None, image, _detection()).map) for k in (0.25, 0.5, 1.0, 1.4)]
    assert gini == sorted(gini, reverse=True)
    assert scaled_prior_name(0.35) == "bbox_prior_s035"
    assert is_control_method("bbox_prior_s035") and is_control_method("random") and not is_control_method("gcame")
    assert base_method_name("bbox_prior_s035") == "bbox_prior"
    with pytest.raises(ValueError):
        get_explainer("bbox_prior_s35", {})


def test_run_evidence_accepts_scaled_priors() -> None:
    run_evidence = _load_script("run_evidence")
    explainers = run_evidence._build_explainers([], ["bbox_prior_s035", "bbox_prior"], {}, False)
    assert set(explainers) == {"bbox_prior_s035", "bbox_prior"}
    with pytest.raises(ValueError):
        run_evidence._build_explainers([], ["bbox_prior_x"], {}, False)


def _metrics(composite: float, sparsity: float) -> dict[str, float]:
    return {"composite": composite, "sparsity": sparsity, "oa": 0.3, "ebpg": 0.5, "oa_box": 0.15, "pg": 1.0}


def _write_run(run_dir: Path, records: list[dict]) -> None:
    (run_dir / "records").mkdir(parents=True)
    (run_dir / "records" / "shard_000.jsonl").write_text("\n".join(json.dumps(record) for record in records) + "\n")


def test_matched_prior_audit(tmp_path: Path, monkeypatch) -> None:
    import yaml

    rng = np.random.default_rng(0)
    test_records, prior_records = [], []
    for index in range(40):
        test_dets, prior_dets = [], []
        for det_id in range(3):
            base = {
                "det_id": det_id, "bbox": [10 * det_id, 10, 10 * det_id + 8, 18], "class_id": 0, "class_name": "person",
                "confidence": 0.8, "explained": True, "match": {"label": "tp" if rng.random() < 0.6 else "fp"},
            }
            noise = float(rng.normal(0, 0.01))
            # A sharp method (Gini 0.95) beats the default prior (0.83) but not a sharp prior (0.94).
            test_dets.append({**base, "methods": {"gcame": _metrics(0.74 + noise, 0.95), "gradcam": _metrics(0.54 + noise, 0.85),
                                                  "bbox_prior": _metrics(0.72, 0.83)}})
            prior_dets.append({**base, "methods": {"bbox_prior_s035": _metrics(0.76, 0.94), "bbox_prior_s100": _metrics(0.72, 0.83),
                                                   "bbox_prior_s140": _metrics(0.65, 0.70)}})
        test_records.append({"image": f"{index:06d}.jpg", "detector": "toy", "detections": test_dets})
        prior_records.append({"image": f"{index:06d}.jpg", "detector": "toy", "detections": prior_dets})
    test_dir, prior_dir = tmp_path / "coco_test_toy", tmp_path / "e1priors_test_toy"
    _write_run(test_dir, test_records)
    _write_run(prior_dir, prior_records)

    analysis = _load_script("analyze_matched_priors")
    summary = analysis.analyze_pair(test_dir, prior_dir, {"bootstrap_samples": 50, "noninferiority_margin": 0.01}, tmp_path, np.random.default_rng(0))
    assert summary["replication"]["box_or_class_mismatch"] == 0
    assert summary["replication"]["bbox_prior_vs_s100_max_abs_diff"] == 0.0
    matched = {row["method"]: row for row in summary["matched"]}
    assert matched["gcame"]["matched_prior"] == "bbox_prior_s035" and matched["gcame"]["holds"]
    assert matched["gradcam"]["matched_prior"] == "bbox_prior_s100" and matched["gradcam"]["prior_superior"]
    assert summary["E1a_verdict"] == "SUPPORTED" and summary["best_prior"]["prior"] == "bbox_prior_s035"
    calibrated = {row["method"]: row for row in summary["calibrated"]}
    assert calibrated["gcame"]["calibrated_composite"] == pytest.approx(-0.02, abs=0.01)

    config = tmp_path / "e1.yaml"
    config.write_text(yaml.safe_dump({"analysis": {"bootstrap_samples": 20, "noninferiority_margin": 0.01}}))
    monkeypatch.setattr(sys, "argv", ["analyze_matched_priors.py", "--run-pairs", f"{test_dir}:{prior_dir}",
                                      "--config", str(config), "--output-dir", str(tmp_path / "e1")])
    analysis.main()
    assert "**E1a**" in (tmp_path / "e1" / "report.md").read_text()
    assert (tmp_path / "e1" / "matched_coco_test_toy.csv").exists()


def test_map_verifier_crops() -> None:
    verifier = _load_script("analyze_map_verifier")
    x0, y0, side = verifier._square_window([10, 20, 30, 30], 2.0)
    assert (side, x0, y0) == (40, 0, 5)
    image = np.arange(12, dtype=np.uint8).reshape(3, 4)
    crop = verifier._crop(image, -1, -1, 3, 99)
    assert crop[0, 0] == 99 and crop[1, 1] == image[0, 0] and crop[2, 2] == image[1, 1]
    assert verifier._full_resolution(np.ones((2, 3), np.float16), 3, 5).shape == (3, 5)


def test_map_verifier_end_to_end(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("cv2")
    rng = np.random.default_rng(0)
    images_dir, run_dir = tmp_path / "images", tmp_path / "run"
    (run_dir / "maps").mkdir(parents=True)
    rows = []
    for index in range(24):
        name = f"{index:06d}.jpg"
        save_image((rng.random((48, 64, 3)) * 255).astype(np.uint8), str(images_dir / name))
        maps = {}
        for det_id in range(2):
            box = [8 + 24 * det_id, 10, 28 + 24 * det_id, 34]
            is_tp = bool(rng.random() < 0.5)
            for method in ("gcame", "lime"):
                saliency = np.zeros((48, 64), dtype=np.float32)
                if is_tp:
                    saliency[box[1] : box[3], box[0] : box[2]] = 1.0
                maps[f"det{det_id}__{method}"] = saliency[::2, ::2].astype(np.float16)
            rows.append({"image": name, "det_id": det_id, "bbox": json.dumps(box), "is_tp": is_tp,
                         "p_S2_baseline": float(np.clip(0.5 + rng.normal(0, 0.05), 0.01, 0.99))})
        np.savez_compressed(run_dir / "maps" / f"{index:06d}.npz", **maps)
    predictions = tmp_path / "g1_predictions_run.csv"
    pd.DataFrame(rows).to_csv(predictions, index=False)

    verifier = _load_script("analyze_map_verifier")
    monkeypatch.setattr(sys, "argv", ["analyze_map_verifier.py", "--run-dir", str(run_dir), "--predictions", str(predictions),
                                      "--images-dir", str(images_dir), "--output-dir", str(tmp_path / "out"), "--methods", "gcame,lime",
                                      "--size", "16", "--epochs", "2", "--folds", "3", "--bootstrap-samples", "20", "--device", "cpu"])
    verifier.main()
    summary = json.loads((tmp_path / "out" / "summary_run.json").read_text())
    assert summary["num_detections"] == 48 and summary["missing_maps"] == 0
    assert summary["claims"][0]["claim"] == "E3_maps_add_no_signal" and "S2+RM - S2+R" in summary["bootstrap"]
    out = pd.read_csv(tmp_path / "out" / "e3_predictions_run.csv")
    assert {"p_S2_baseline", "p_R", "p_RM", "p_M", "p_S2+RM"} <= set(out.columns)
    data = np.load(next((tmp_path / "out").glob("dataset_*.npz")))
    tp_rows = np.flatnonzero(pd.DataFrame(rows)["is_tp"].to_numpy())
    assert float(data["maps"][tp_rows].astype(np.float32).mean()) > 0.1 and float(data["mask"].mean()) > 0.1
