"""Stage 2 of the evidence pipeline: VLM verification of boxes and salient regions (GPU).

Reads the records and saved maps written by ``scripts/run_evidence.py`` and asks
the VLM, for every explained detection, P(yes) that

- the box crop shows the detected class (``box_p_yes``), and
- the salient pixels of each configured map show it (``sal_<method>_p_yes``).

Detector confidence is never shown to the VLM. Results go to
``<run_dir>/vlm/shard_XXX.jsonl`` (one line per detection); the loader in
``src/evidence_io.py`` joins them as ``vlm__*`` columns. Sharded and resumable.

Example:
    python scripts/run_vlm_verification.py \
        --run-dir results/evidence/coco_test_yolox-s \
        --images-dir data/coco/val2017 --num-shards 4 --shard-index 0
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detector import Detection
from src.evidence_io import iter_image_records
from src.provenance import write_manifest
from src.utils import load_image, setup_logging
from src.vlm_judge import VLMJudge

logger = logging.getLogger("run_vlm_verification")


def _as_project_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def _done_keys(path: Path) -> set[tuple[str, int]]:
    done: set[tuple[str, int]] = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add((str(record["image"]), int(record["det_id"])))
    return done


def _detection_from_record(det: dict[str, Any]) -> Detection:
    x1, y1, x2, y2 = [int(value) for value in det["bbox"]]
    return Detection(
        bbox=[x1, y1, x2, y2],
        class_id=int(det["class_id"]),
        class_name=str(det["class_name"]),
        confidence=float(det["confidence"]),
        area=max(0, x2 - x1) * max(0, y2 - y1),
        relative_size=str(det.get("relative_size", "medium")),
        detection_id=int(det["det_id"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--vlm-config", default="config/vlm_config.yaml")
    parser.add_argument("--evidence-config", default="config/evidence_config.yaml")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    setup_logging(args.log_level)

    run_dir = _as_project_path(args.run_dir)
    images_dir = _as_project_path(args.images_dir)
    with _as_project_path(args.evidence_config).open("r", encoding="utf-8") as handle:
        vlm_cfg = dict((yaml.safe_load(handle) or {}).get("vlm_verification", {}))
    view_kwargs = {
        "box_context_scale": float(vlm_cfg.get("box_context_scale", 1.3)),
        "salient_box_multiple": float(vlm_cfg.get("salient_box_multiple", 2.0)),
        "min_image_fraction": float(vlm_cfg.get("min_image_fraction", 0.01)),
    }
    saliency_methods = list(vlm_cfg.get("saliency_methods", []))
    do_box = bool(vlm_cfg.get("box", True))

    records = sorted(iter_image_records([run_dir]), key=lambda record: record["image"])
    shard_records = records[args.shard_index :: args.num_shards]
    shard_tag = f"shard_{args.shard_index:03d}"
    out_path = run_dir / "vlm" / f"{shard_tag}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _done_keys(out_path)

    write_manifest(
        run_dir / "vlm" / "manifests",
        entry_point="scripts/run_vlm_verification.py",
        args=vars(args),
        config_paths={"vlm": args.vlm_config, "evidence": args.evidence_config},
        image_names=[record["image"] for record in shard_records],
        extra={"view_kwargs": view_kwargs, "saliency_methods": saliency_methods, "box": do_box},
        file_name=f"{shard_tag}.json",
    )

    judge = VLMJudge(config_path=str(_as_project_path(args.vlm_config)))
    judge.load_model()
    try:
        with out_path.open("a", encoding="utf-8") as handle:
            for record in tqdm(shard_records, desc=f"VLM {shard_tag}", unit="image"):
                pending = [
                    det
                    for det in record.get("detections", [])
                    if det.get("explained") and (record["image"], int(det["det_id"])) not in done
                ]
                if not pending:
                    continue
                image = load_image(str(images_dir / record["image"]))
                maps_path = run_dir / "maps" / f"{Path(record['image']).stem}.npz"
                saved = np.load(maps_path) if maps_path.exists() else None
                for det in pending:
                    detection = _detection_from_record(det)
                    features: dict[str, float] = {}
                    total_time = 0.0
                    if do_box:
                        result = judge.verify_region(image, detection, None, **view_kwargs)
                        features["box_p_yes"] = result["p_yes"]
                        features["box_answer_mass"] = result["answer_mass"]
                        total_time += result["vlm_time"]
                    for method in saliency_methods:
                        key = f"det{detection.detection_id}__{method}"
                        if saved is None or key not in saved.files:
                            continue
                        small = saved[key].astype(np.float32)
                        full = cv2.resize(small, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
                        result = judge.verify_region(image, detection, full, **view_kwargs)
                        features[f"sal_{method}_p_yes"] = result["p_yes"]
                        features[f"sal_{method}_answer_mass"] = result["answer_mass"]
                        total_time += result["vlm_time"]
                    features["time"] = total_time
                    handle.write(
                        json.dumps({"image": record["image"], "det_id": detection.detection_id, "features": features})
                        + "\n"
                    )
                    handle.flush()
    finally:
        judge.unload_model()


if __name__ == "__main__":
    main()
