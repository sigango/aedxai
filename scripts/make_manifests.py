"""Create fixed, disjoint image manifests for the v2 experiments.

Every run then uses ``--image-list <manifest> --split <name>`` instead of the
sorted-prefix image selection of the legacy scripts, so calibration, analysis
and test images never overlap and every result can be traced to its image list.

By default the images used to build the old selector-oracle data (image ids in
``results/xai_selector_training_progress_*.csv``) are excluded, because the
legacy selector, the provisional cascade tau and the old ablation were fitted
on them.

Example:
    python scripts/make_manifests.py \
        --images-dir data/coco/val2017 \
        --annotations data/coco/annotations/instances_val2017.json \
        --splits dev:300,calib:1000,test:2000 \
        --output manifests/coco_val2017_v2.json
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.provenance import image_list_sha256


def _as_project_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def _oracle_image_ids(patterns: list[str]) -> set[int]:
    """Image ids appearing in legacy selector progress CSVs (sample_key = 'image_id:class:bbox')."""
    ids: set[int] = set()
    for pattern in patterns:
        matches = glob.glob(str(_as_project_path(pattern)))
        if not matches:
            # results/ is gitignored, so a fresh clone has no legacy CSVs; failing here
            # prevents manifests that silently keep the v1 oracle images.
            raise SystemExit(
                f"No legacy oracle CSVs match '{pattern}'. Copy results/ from the old runs "
                "(see RESULTS_AND_PLAN.md B3) or pass --exclude-oracle-csvs '' to skip exclusion."
            )
        for path in matches:
            frame = pd.read_csv(path, usecols=["sample_key"])
            ids.update(int(str(key).split(":")[0]) for key in frame["sample_key"])
    return ids


def _parse_splits(value: str) -> list[tuple[str, int]]:
    splits: list[tuple[str, int]] = []
    for item in value.split(","):
        name, _, size = item.partition(":")
        if not name or not size:
            raise ValueError(f"Bad split spec '{item}'; expected name:size")
        splits.append((name.strip(), int(size)))
    return splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--annotations", default=None, help="If given, keep only images with >=1 annotation.")
    parser.add_argument("--splits", default="dev:300,calib:1000,test:2000")
    parser.add_argument(
        "--exclude-oracle-csvs",
        default="results/xai_selector_training_progress_*.csv",
        help="Glob(s), comma separated; '' to disable.",
    )
    parser.add_argument("--exclude-first-n", type=int, default=0, help="Also exclude the first N sorted images.")
    parser.add_argument("--extensions", default=".jpg,.jpeg,.png")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    images_dir = _as_project_path(args.images_dir)
    extensions = {ext.strip().lower() for ext in args.extensions.split(",")}
    names = sorted(path.name for path in images_dir.iterdir() if path.suffix.lower() in extensions)
    excluded: dict[str, int] = {}

    if args.annotations:
        with _as_project_path(args.annotations).open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        annotated_ids = {int(annotation["image_id"]) for annotation in raw.get("annotations", [])}
        annotated = {
            Path(str(image["file_name"])).name for image in raw.get("images", []) if int(image["id"]) in annotated_ids
        }
        before = len(names)
        names = [name for name in names if name in annotated]
        excluded["no_annotations"] = before - len(names)

    if args.exclude_first_n > 0:
        first = set(sorted(names)[: args.exclude_first_n])
        names = [name for name in names if name not in first]
        excluded["first_n"] = len(first)

    patterns = [item.strip() for item in args.exclude_oracle_csvs.split(",") if item.strip()]
    if patterns:
        oracle_ids = _oracle_image_ids(patterns)

        def _image_id(name: str) -> int | None:
            stem = Path(name).stem
            return int(stem) if stem.isdigit() else None

        before = len(names)
        names = [name for name in names if _image_id(name) not in oracle_ids]
        excluded["legacy_oracle_images"] = before - len(names)

    rng = np.random.default_rng(args.seed)
    shuffled = list(rng.permutation(names))
    splits = _parse_splits(args.splits)
    requested = sum(size for _, size in splits)
    if requested > len(shuffled):
        raise ValueError(f"Requested {requested} images but only {len(shuffled)} remain after exclusions.")

    manifest_splits: dict[str, list[str]] = {}
    cursor = 0
    for name, size in splits:
        manifest_splits[name] = sorted(str(item) for item in shuffled[cursor : cursor + size])
        cursor += size

    output = _as_project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "images_dir": str(args.images_dir),
        "annotations": args.annotations,
        "seed": args.seed,
        "excluded": excluded,
        "available_after_exclusion": len(shuffled),
        "splits": manifest_splits,
        "sha256": {name: image_list_sha256(items) for name, items in manifest_splits.items()},
    }
    with output.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=1)
    print(json.dumps({key: value for key, value in manifest.items() if key != "splits"}, indent=1))
    print({name: len(items) for name, items in manifest_splits.items()}, "->", output)


if __name__ == "__main__":
    main()
