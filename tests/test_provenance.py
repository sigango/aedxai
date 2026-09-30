"""Tests for run manifests and image-list manifests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from src.provenance import load_image_list, resolve_image_paths, write_manifest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_write_manifest_records_inputs(tmp_path: Path) -> None:
    config = tmp_path / "cfg.yaml"
    config.write_text("a: 1\n")
    path = write_manifest(
        tmp_path / "run",
        entry_point="test",
        args={"x": Path("p")},
        config_paths={"cfg": config},
        image_names=["b.jpg", "a.jpg"],
        checkpoints={"selector": None},
    )
    manifest = json.loads(path.read_text())
    assert manifest["images"]["count"] == 2 and manifest["images"]["names"] == ["b.jpg", "a.jpg"]
    assert manifest["configs"]["cfg"]["text"] == "a: 1\n" and len(manifest["configs"]["cfg"]["sha256"]) == 64
    assert "commit" in manifest["git"] and manifest["args"]["x"] == "p"


def test_image_list_json_and_txt(tmp_path: Path) -> None:
    (tmp_path / "m.json").write_text(json.dumps({"splits": {"dev": ["1.jpg"], "test": ["2.jpg", "3.jpg"]}}))
    assert load_image_list(tmp_path / "m.json", split="test") == ["2.jpg", "3.jpg"]
    with pytest.raises(ValueError):
        load_image_list(tmp_path / "m.json")
    (tmp_path / "list.txt").write_text("# header\n1.jpg\n\n2.jpg\n")
    assert load_image_list(tmp_path / "list.txt") == ["1.jpg", "2.jpg"]


def test_resolve_image_paths_checks_existence(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    for name in ("b.jpg", "a.jpg", "c.png"):
        (images / name).write_bytes(b"")
    assert [p.name for p in resolve_image_paths(images, extensions=(".jpg",))] == ["a.jpg", "b.jpg"]
    (tmp_path / "list.txt").write_text("b.jpg\nmissing.jpg\n")
    with pytest.raises(FileNotFoundError):
        resolve_image_paths(images, image_list=tmp_path / "list.txt")


def test_make_manifests_disjoint_splits(tmp_path: Path, monkeypatch) -> None:
    import importlib.util

    images = tmp_path / "images"
    images.mkdir()
    for index in range(30):
        (images / f"{index:012d}.jpg").write_bytes(b"")
    progress = tmp_path / "progress.csv"
    progress.write_text("sample_key\n3:0:1-2-3-4\n4:1:1-2-3-4\n")
    spec = importlib.util.spec_from_file_location("make_manifests", PROJECT_ROOT / "scripts" / "make_manifests.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path / "manifest.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["make_manifests.py", "--images-dir", str(images), "--splits", "dev:5,test:10",
         "--exclude-oracle-csvs", str(progress), "--output", str(output)],
    )
    module.main()
    manifest = json.loads(output.read_text())
    dev, test = set(manifest["splits"]["dev"]), set(manifest["splits"]["test"])
    assert len(dev) == 5 and len(test) == 10 and not dev & test
    assert "000000000003.jpg" not in dev | test and manifest["excluded"]["legacy_oracle_images"] == 2
