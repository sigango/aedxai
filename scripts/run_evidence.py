"""Collect per-detection explanation evidence with ground-truth labels (GPU stage).

For every image in a manifest this script:

1. runs the detector once at fixed thresholds (no feedback loop, no VLM);
2. matches detections to COCO-format ground truth (TP / FP / FP type);
3. computes the cheap baseline features for every detection: post-NMS output
   statistics, pre-NMS candidate statistics (one extra forward with NMS off) and
   test-time-augmentation stability (one forward per TTA view);
4. for the top-``max_explained_per_image`` detections, runs every configured
   explainer and control, scores each map with ``AutoEvaluator.evaluate_evidence``
   (paper-facing metrics + curves + box-relative protocol + GT-box localization),
   and computes gradient, map and agreement features;
5. appends one JSON line per image to ``<run_dir>/records/shard_XXX.jsonl``.

It is sharded (``--num-shards/--shard-index``) and resumable: images already in
a shard file are skipped, so a pre-empted cluster job can simply be resubmitted.

Example:
    python scripts/run_evidence.py \
        --images-dir data/coco/val2017 \
        --annotations data/coco/annotations/instances_val2017.json \
        --image-list manifests/coco_val2017_v2.json --split test \
        --detector-model yolox-s --run-name coco_test_yolox-s \
        --num-shards 8 --shard-index 0
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detector import DetectorWrapper
from src.evaluator import AutoEvaluator
from src.evidence import (
    agreement_features,
    candidate_arrays,
    gradient_features,
    map_features,
    output_features,
    prenms_features,
    run_tta,
    tta_features,
)
from src.gt_matching import AnnotationIndex, match_detections
from src.provenance import resolve_image_paths, write_manifest
from src.utils import load_image, set_seed, setup_logging
from src.xai_methods import CAM_METHODS, base_method_name, get_explainer, shuffle_saliency

logger = logging.getLogger("run_evidence")

CONTROL_METHODS = {"bbox_prior", "center_prior", "random"}


def _as_project_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _done_images(shard_path: Path) -> set[str]:
    """Images already written successfully to this shard (resume support).

    Images whose record holds an ``error`` are retried on the next submission;
    the loader keeps the last successful record per image.
    """
    done: set[str] = set()
    if not shard_path.exists():
        return done
    with shard_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping a corrupt line in %s (likely an interrupted write).", shard_path)
                continue
            if "error" not in record and "image" in record:
                done.add(str(record["image"]))
    return done


def _build_explainers(
    methods: list[str],
    controls: list[str],
    xai_config: dict[str, Any],
    dclose_final_masks: bool,
) -> dict[str, Any]:
    method_sections = dict(xai_config.get("xai", {}).get("methods", {}))
    explainers: dict[str, Any] = {}
    for name in methods:
        section = dict(method_sections.get(base_method_name(name), {}))
        if base_method_name(name) == "dclose":
            section["use_final_masks"] = bool(dclose_final_masks)
        explainers[name] = get_explainer(name, section)
    for name in controls:
        if name not in CONTROL_METHODS:
            raise ValueError(f"Unknown control '{name}'. Available: {sorted(CONTROL_METHODS)}")
        explainers[name] = get_explainer(name, {})
    return explainers


def _detection_seed(image_name: str, detection_id: int) -> int:
    return zlib.crc32(f"{image_name}:{detection_id}".encode("utf-8"))


def _method_record(evaluation: dict[str, Any], saliency: Any, box_protocol: bool) -> dict[str, Any]:
    keys = [
        "pg",
        "ebpg",
        "oa",
        "insertion_auc",
        "deletion_auc",
        "sparsity",
        "composite",
        "pg_gt",
        "ebpg_gt",
        "eval_time",
        "insertion_curve",
        "deletion_curve",
    ]
    if box_protocol:
        keys += [
            "oa_box",
            "insertion_auc_box",
            "deletion_auc_box",
            "composite_box",
            "insertion_curve_box",
            "deletion_curve_box",
        ]
    record = {key: evaluation[key] for key in keys if key in evaluation}
    record["explain_time"] = float(getattr(saliency, "computation_time", 0.0))
    record["is_fallback"] = bool(getattr(saliency, "is_fallback", False))
    record["fallback_reason"] = getattr(saliency, "fallback_reason", None)
    return record


def process_image(
    image_path: Path,
    *,
    detector: DetectorWrapper,
    evaluator: AutoEvaluator,
    explainers: dict[str, Any],
    annotations: AnnotationIndex,
    settings: dict[str, Any],
    maps_dir: Path | None,
) -> dict[str, Any]:
    """Run detection, GT matching, explanation and evaluation for one image."""
    t_image = time.time()
    image = load_image(str(image_path))
    height, width = image.shape[:2]
    gt_entry = annotations.get(image_path.name)
    gt_boxes = list(gt_entry.boxes) if gt_entry is not None else []

    t_detect = time.time()
    detections = detector.detect(image, conf_thresh=settings["conf_thresh"])
    detect_time = time.time() - t_detect
    model = detector.get_model()

    prenms_cfg = dict(settings.get("prenms") or {})
    candidates = None
    t_prenms = time.time()
    if prenms_cfg.get("enabled") and hasattr(detector, "detect_candidates"):
        candidates = candidate_arrays(
            detector.detect_candidates(
                image,
                conf_thresh=float(prenms_cfg.get("conf_thresh", 0.01)),
                max_candidates=int(prenms_cfg.get("max_candidates", 3000)),
            )
        )
    prenms_time = time.time() - t_prenms
    tta_cfg = dict(settings.get("tta") or {})
    tta_outputs = None
    t_tta = time.time()
    if tta_cfg.get("enabled"):
        tta_outputs = run_tta(
            detector,
            image,
            views=list(tta_cfg.get("views", ["hflip", "shrink", "shift", "dark"])),
            conf_thresh=float(tta_cfg.get("conf_thresh", 0.05)),
        )
    tta_time = time.time() - t_tta

    matches = match_detections(
        boxes=[det.bbox for det in detections],
        class_ids=[det.class_id for det in detections],
        scores=[det.confidence for det in detections],
        gt_boxes=gt_boxes,
        iou_threshold=settings["gt_iou_threshold"],
    )
    explained_ids = {
        det.detection_id
        for det in sorted(detections, key=lambda item: item.confidence, reverse=True)[
            : settings["max_explained_per_image"]
        ]
    }
    needs_target_layer = settings["gradient_features"] or any(name in CAM_METHODS for name in explainers)
    target_layer = detector.get_target_layer() if needs_target_layer else None

    saved_maps: dict[str, np.ndarray] = {}
    detection_records: list[dict[str, Any]] = []
    for detection, match in zip(detections, matches):
        record: dict[str, Any] = {
            "det_id": int(detection.detection_id),
            "bbox": [int(value) for value in detection.bbox],
            "class_id": int(detection.class_id),
            "class_name": detection.class_name,
            "confidence": float(detection.confidence),
            "relative_size": detection.relative_size,
            "explained": detection.detection_id in explained_ids,
            "match": {
                "label": match.label,
                "iou_gt": float(match.iou_gt),
                "gt_bbox": match.gt_bbox,
                "fp_type": match.fp_type,
                "best_iou_same_class": float(match.best_iou_same_class),
                "best_iou_any_class": float(match.best_iou_any_class),
            },
            "output": output_features(detection, detections, image),
        }
        if candidates is not None:
            record["prenms"] = prenms_features(detection, *candidates)
        if tta_outputs is not None:
            record["tta"] = tta_features(detection, tta_outputs)
        if not record["explained"]:
            detection_records.append(record)
            continue

        if settings["gradient_features"]:
            try:
                record["gradient"] = gradient_features(model, image, detection, target_layer)
            except Exception as exc:  # noqa: BLE001 - keep the image; mark the feature missing
                logger.warning("Gradient features failed for %s det %d: %s", image_path.name, detection.detection_id, exc)
                record["gradient"] = None

        gt_bbox = [int(round(value)) for value in match.gt_bbox] if match.gt_bbox else None
        maps: dict[str, np.ndarray] = {}
        method_records: dict[str, Any] = {}
        for name, explainer in explainers.items():
            saliency = explainer.explain(
                model=model,
                image=image,
                detection=detection,
                target_layer=target_layer if name in CAM_METHODS else None,
            )
            evaluation = evaluator.evaluate_evidence(
                saliency.map,
                detection,
                model,
                image,
                gt_bbox=gt_bbox,
                include_box_protocol=settings["box_protocol"],
            )
            method_record = _method_record(evaluation, saliency, settings["box_protocol"])
            method_record["map_features"] = map_features(saliency.map, detection.bbox)
            method_records[name] = method_record
            maps[name] = saliency.map

        for name in settings["shuffled_of"]:
            if name not in maps:
                continue
            shuffled = shuffle_saliency(maps[name], seed=_detection_seed(image_path.name, detection.detection_id))
            evaluation = evaluator.evaluate_evidence(
                shuffled,
                detection,
                model,
                image,
                gt_bbox=gt_bbox,
                include_box_protocol=settings["box_protocol"],
            )
            method_records[f"shuffled_{name}"] = _method_record(evaluation, None, settings["box_protocol"])

        record["methods"] = method_records
        record["agreement"] = {
            f"{first}|{second}": agreement_features(maps[first], maps[second], detection.bbox)
            for first, second in settings["agreement_pairs"]
            if first in maps and second in maps
        }
        step = max(1, int(settings["map_downsample"]))
        for name in settings["save_maps_for"]:
            if name in maps:
                saved_maps[f"det{detection.detection_id}__{name}"] = maps[name][::step, ::step].astype(np.float16)
        detection_records.append(record)

    if maps_dir is not None and saved_maps:
        maps_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(maps_dir / f"{image_path.stem}.npz", **saved_maps)

    return {
        "image": image_path.name,
        "image_id": int(gt_entry.image_id) if gt_entry is not None else None,
        "width": int(width),
        "height": int(height),
        "attributes": dict(gt_entry.attributes) if gt_entry is not None else {},
        "has_annotations": gt_entry is not None,
        "gt": [
            {"bbox": [float(v) for v in gt.bbox], "class_id": int(gt.class_id), "iscrowd": bool(gt.iscrowd)}
            for gt in gt_boxes
        ],
        "detector": detector.model_name,
        "detect_time": float(detect_time),
        "prenms_time": float(prenms_time),
        "tta_time": float(tta_time),
        "image_time": float(time.time() - t_image),
        "detections": detection_records,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--annotations", required=True, help="COCO-format annotation JSON.")
    parser.add_argument("--image-list", default=None, help="Manifest from scripts/make_manifests.py (or .txt list).")
    parser.add_argument("--split", default=None, help="Split name inside the manifest (e.g. dev, calib, test).")
    parser.add_argument("--num-images", type=int, default=0, help="Optional cap after manifest selection (0 = all).")
    parser.add_argument("--detector-model", required=True, choices=["yolox-s", "fasterrcnn_resnet50_fpn_v2"])
    parser.add_argument("--run-name", required=True, help="Directory name under --output-root.")
    parser.add_argument("--output-root", default="results/evidence")
    parser.add_argument("--evidence-config", default="config/evidence_config.yaml")
    parser.add_argument("--detector-config", default="config/detector_config.yaml")
    parser.add_argument("--xai-config", default="config/xai_config.yaml")
    parser.add_argument("--eval-config", default="config/eval_config.yaml")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--methods", default=None, help="Comma list overriding evidence.methods.")
    parser.add_argument("--controls", default=None, help="Comma list overriding evidence.controls ('' = none).")
    parser.add_argument("--dclose-final-masks", action="store_true", help="Use 2000 D-CLOSE masks.")
    parser.add_argument("--max-explained", type=int, default=None, help="Override evidence.max_explained_per_image.")
    parser.add_argument("--no-maps", action="store_true", help="Do not save saliency maps.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    setup_logging(args.log_level)
    set_seed(args.seed)
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, --num-shards)")

    evidence_config_path = _as_project_path(args.evidence_config)
    evidence_cfg = dict(_load_yaml(evidence_config_path).get("evidence", {}))
    detector_cfg = dict(_load_yaml(_as_project_path(args.detector_config)).get("detector", {}))
    eval_raw = _load_yaml(_as_project_path(args.eval_config))
    eval_cfg = dict(eval_raw.get("evaluation", eval_raw))
    eval_cfg.setdefault("metrics", {}).setdefault("over_all", {})["batch_size"] = int(
        evidence_cfg.get("evaluator_batch_size", 1)
    )

    methods = (
        [item.strip() for item in args.methods.split(",") if item.strip()]
        if args.methods is not None
        else list(evidence_cfg.get("methods", []))
    )
    controls = (
        [item.strip() for item in args.controls.split(",") if item.strip()]
        if args.controls is not None
        else list(evidence_cfg.get("controls", []))
    )
    settings = {
        "conf_thresh": (
            float(evidence_cfg["conf_thresh"])
            if evidence_cfg.get("conf_thresh") is not None
            else float(detector_cfg.get("conf_thresh", 0.25))
        ),
        "max_explained_per_image": int(
            args.max_explained if args.max_explained is not None else evidence_cfg.get("max_explained_per_image", 20)
        ),
        "gradient_features": bool(evidence_cfg.get("gradient_features", True)),
        "gt_iou_threshold": float(evidence_cfg.get("gt_iou_threshold", 0.5)),
        "box_protocol": bool(evidence_cfg.get("box_protocol", True)),
        "shuffled_of": list(evidence_cfg.get("shuffled_of", [])),
        "agreement_pairs": [tuple(pair) for pair in evidence_cfg.get("agreement_pairs", [])],
        "save_maps_for": [] if args.no_maps else list(evidence_cfg.get("save_maps_for", [])),
        "map_downsample": int(evidence_cfg.get("map_downsample", 2)),
        "prenms": dict(evidence_cfg.get("prenms") or {"enabled": False}),
        "tta": dict(evidence_cfg.get("tta") or {"enabled": False}),
    }
    dclose_final = bool(args.dclose_final_masks or evidence_cfg.get("dclose_final_masks", False))

    image_paths = resolve_image_paths(
        _as_project_path(args.images_dir),
        image_list=_as_project_path(args.image_list) if args.image_list else None,
        split=args.split,
        num_images=args.num_images or None,
    )
    shard_paths = image_paths[args.shard_index :: args.num_shards]

    run_dir = _as_project_path(args.output_root) / args.run_name
    shard_tag = f"shard_{args.shard_index:03d}"
    records_path = run_dir / "records" / f"{shard_tag}.jsonl"
    records_path.parent.mkdir(parents=True, exist_ok=True)
    maps_dir = None if not settings["save_maps_for"] else run_dir / "maps"
    done = _done_images(records_path)
    todo = [path for path in shard_paths if path.name not in done]
    logger.info(
        "Run %s %s: %d images in shard, %d already done, %d to process.",
        args.run_name,
        shard_tag,
        len(shard_paths),
        len(done),
        len(todo),
    )

    write_manifest(
        run_dir / "manifests",
        entry_point="scripts/run_evidence.py",
        args={**vars(args), "methods": methods, "controls": controls, "dclose_final_masks": dclose_final},
        config_paths={
            "evidence": args.evidence_config,
            "detector": args.detector_config,
            "xai": args.xai_config,
            "eval": args.eval_config,
        },
        image_names=[path.name for path in shard_paths],
        extra={"settings": {**settings, "agreement_pairs": [list(p) for p in settings["agreement_pairs"]]}},
        file_name=f"{shard_tag}.json",
    )

    annotations = AnnotationIndex.from_coco_json(_as_project_path(args.annotations))
    missing_gt = sum(1 for path in shard_paths if path.name not in annotations)
    if missing_gt:
        logger.warning("%d/%d shard images have no annotations; they are kept with an empty GT list.", missing_gt, len(shard_paths))

    detector = DetectorWrapper(args.detector_model, config_path=str(_as_project_path(args.detector_config)))
    detector.load_model()
    evaluator = AutoEvaluator(eval_cfg)
    explainers = _build_explainers(methods, controls, _load_yaml(_as_project_path(args.xai_config)), dclose_final)

    try:
        with records_path.open("a", encoding="utf-8") as handle:
            for image_path in tqdm(todo, desc=f"{args.run_name}:{shard_tag}", unit="image"):
                try:
                    record = process_image(
                        image_path,
                        detector=detector,
                        evaluator=evaluator,
                        explainers=explainers,
                        annotations=annotations,
                        settings=settings,
                        maps_dir=maps_dir,
                    )
                except Exception as exc:  # noqa: BLE001 - record and move on
                    logger.warning("Image %s failed: %s", image_path.name, exc, exc_info=True)
                    record = {"image": image_path.name, "detector": args.detector_model, "error": repr(exc)}
                handle.write(json.dumps(record) + "\n")
                handle.flush()
    finally:
        detector.unload_model()
        gc.collect()
    logger.info("Wrote %s", records_path)


if __name__ == "__main__":
    main()
