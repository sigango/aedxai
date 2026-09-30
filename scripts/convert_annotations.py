"""Convert cross-domain annotations to COCO-format JSON with COCO-80 category names.

The evidence runner reads only COCO-format files (``src/gt_matching.py``), and
detectors predict COCO-80 classes, so every dataset is mapped onto those names.
Classes without a COCO counterpart are dropped; ignore/difficult regions become
crowd boxes so detections on them are neither TP nor FP.

Supported formats:
    voc       Pascal VOC XML directory (difficult objects -> crowd)
    bdd100k   BDD100K detection JSON (list of frames with "labels"/"box2d");
              image attributes (weather, timeofday, scene) are kept for subgroups
    visdrone  VisDrone-DET txt directory (category 0 "ignored regions" -> ignore)

Only images present in ``--images-dir`` are written.

Example:
    python scripts/convert_annotations.py --format bdd100k \
        --source data/bdd100k/labels/det_20/det_val.json \
        --images-dir data/bdd100k/images/100k/val \
        --output data/bdd100k/annotations_coco80_val.json
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gt_matching import IGNORE_CATEGORY_NAME
from src.utils import COCO_CLASSES

VOC_TO_COCO = {
    "aeroplane": "airplane",
    "bicycle": "bicycle",
    "bird": "bird",
    "boat": "boat",
    "bottle": "bottle",
    "bus": "bus",
    "car": "car",
    "cat": "cat",
    "chair": "chair",
    "cow": "cow",
    "diningtable": "dining table",
    "dog": "dog",
    "horse": "horse",
    "motorbike": "motorcycle",
    "person": "person",
    "pottedplant": "potted plant",
    "sheep": "sheep",
    "sofa": "couch",
    "train": "train",
    "tvmonitor": "tv",
}

BDD_TO_COCO = {
    "pedestrian": "person",
    "person": "person",
    "rider": "person",
    "car": "car",
    "bus": "bus",
    "truck": "truck",
    "train": "train",
    "bicycle": "bicycle",
    "bike": "bicycle",
    "motorcycle": "motorcycle",
    "motor": "motorcycle",
    "traffic light": "traffic light",
}

VISDRONE_TO_COCO = {
    1: "person",  # pedestrian
    2: "person",  # people
    3: "bicycle",
    4: "car",
    5: "car",  # van
    6: "truck",
    9: "bus",
    10: "motorcycle",
}


class _CocoWriter:
    def __init__(self) -> None:
        self.images: list[dict[str, Any]] = []
        self.annotations: list[dict[str, Any]] = []
        self.category_ids = {name: index + 1 for index, name in enumerate(COCO_CLASSES)}
        self.category_ids[IGNORE_CATEGORY_NAME] = 0

    def add_image(self, file_name: str, width: int, height: int, attributes: dict[str, Any] | None = None) -> int:
        image_id = len(self.images) + 1
        entry = {"id": image_id, "file_name": file_name, "width": int(width), "height": int(height)}
        if attributes:
            entry["attributes"] = attributes
        self.images.append(entry)
        return image_id

    def add_box(self, image_id: int, name: str, x1: float, y1: float, x2: float, y2: float, iscrowd: bool) -> None:
        width, height = max(0.0, x2 - x1), max(0.0, y2 - y1)
        if width <= 0 or height <= 0:
            return
        self.annotations.append(
            {
                "id": len(self.annotations) + 1,
                "image_id": image_id,
                "category_id": self.category_ids[name],
                "bbox": [float(x1), float(y1), float(width), float(height)],
                "area": float(width * height),
                "iscrowd": int(bool(iscrowd)),
            }
        )

    def dump(self, output: Path, source: str) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        categories = [{"id": cid, "name": name} for name, cid in sorted(self.category_ids.items(), key=lambda x: x[1])]
        with output.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "info": {"source": source, "converted_to": "COCO-80 names"},
                    "images": self.images,
                    "annotations": self.annotations,
                    "categories": categories,
                },
                handle,
            )
        print(f"Wrote {len(self.images)} images / {len(self.annotations)} boxes -> {output}")


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def convert_voc(source: Path, images_dir: Path, writer: _CocoWriter) -> None:
    for image_path in sorted(images_dir.glob("*.jpg")):
        xml_path = source / f"{image_path.stem}.xml"
        if not xml_path.exists():
            continue
        root = ET.parse(xml_path).getroot()
        size = root.find("size")
        if size is not None:
            width, height = int(size.findtext("width", "0")), int(size.findtext("height", "0"))
        else:
            width, height = _image_size(image_path)
        image_id = writer.add_image(image_path.name, width, height)
        for obj in root.findall("object"):
            name = VOC_TO_COCO.get(str(obj.findtext("name", "")).strip().lower())
            box = obj.find("bndbox")
            if name is None or box is None:
                continue
            coords = [float(box.findtext(tag, "0")) for tag in ("xmin", "ymin", "xmax", "ymax")]
            difficult = str(obj.findtext("difficult", "0")).strip() == "1"
            writer.add_box(image_id, name, coords[0] - 1, coords[1] - 1, coords[2], coords[3], iscrowd=difficult)


def convert_bdd100k(source: Path, images_dir: Path, writer: _CocoWriter) -> None:
    with source.open("r", encoding="utf-8") as handle:
        frames = json.load(handle)
    available = {path.name for path in images_dir.glob("*.jpg")}
    for frame in frames:
        name = Path(str(frame["name"])).name
        if name not in available:
            continue
        width, height = _image_size(images_dir / name)
        attributes = {key: frame.get("attributes", {}).get(key) for key in ("weather", "timeofday", "scene")}
        image_id = writer.add_image(name, width, height, attributes)
        for label in frame.get("labels") or []:
            coco_name = BDD_TO_COCO.get(str(label.get("category", "")).strip().lower())
            box = label.get("box2d")
            if coco_name is None or not box:
                continue
            writer.add_box(image_id, coco_name, box["x1"], box["y1"], box["x2"], box["y2"], iscrowd=False)


def convert_visdrone(source: Path, images_dir: Path, writer: _CocoWriter) -> None:
    for image_path in sorted(images_dir.glob("*.jpg")):
        txt_path = source / f"{image_path.stem}.txt"
        if not txt_path.exists():
            continue
        width, height = _image_size(image_path)
        image_id = writer.add_image(image_path.name, width, height)
        with txt_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                parts = [part for part in line.strip().split(",") if part != ""]
                if len(parts) < 6:
                    continue
                x, y, w, h = (float(value) for value in parts[:4])
                category = int(parts[5])
                if category == 0:
                    writer.add_box(image_id, IGNORE_CATEGORY_NAME, x, y, x + w, y + h, iscrowd=True)
                elif category in VISDRONE_TO_COCO:
                    writer.add_box(image_id, VISDRONE_TO_COCO[category], x, y, x + w, y + h, iscrowd=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--format", required=True, choices=["voc", "bdd100k", "visdrone"])
    parser.add_argument("--source", required=True, help="VOC Annotations dir, BDD100K JSON, or VisDrone annotations dir.")
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    writer = _CocoWriter()
    source, images_dir = Path(args.source), Path(args.images_dir)
    {"voc": convert_voc, "bdd100k": convert_bdd100k, "visdrone": convert_visdrone}[args.format](
        source, images_dir, writer
    )
    writer.dump(Path(args.output), source=f"{args.format}:{source}")


if __name__ == "__main__":
    main()
