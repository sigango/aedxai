"""Run manifests and image lists so every result file can be traced to its inputs."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run_git(args: Sequence[str]) -> str | None:
    """Return stripped git output, or None when git or the repo is unavailable."""
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip()


def git_state() -> dict[str, Any]:
    """Commit SHA, dirty flag, and a hash of the uncommitted diff."""
    sha = _run_git(["rev-parse", "HEAD"])
    diff = _run_git(["diff", "HEAD"]) or ""
    status = _run_git(["status", "--porcelain"]) or ""
    return {
        "commit": sha,
        "dirty": bool(status),
        "diff_sha256": hashlib.sha256(diff.encode("utf-8")).hexdigest() if diff else None,
    }


def file_sha256(path: str | Path, chunk_size: int = 1 << 20) -> str | None:
    """SHA-256 of a file, or None if it does not exist."""
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment_info() -> dict[str, Any]:
    """Python, platform, and key library versions (torch/CUDA when importable)."""
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    for module_name in ("numpy", "torch", "torchvision", "transformers", "sklearn"):
        try:
            module = __import__(module_name)
            info[module_name] = getattr(module, "__version__", "unknown")
        except Exception:  # noqa: BLE001 - optional dependency probe
            info[module_name] = None
    try:
        import torch

        info["cuda_available"] = bool(torch.cuda.is_available())
        info["torch_cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        pass
    return info


def config_snapshot(config_paths: Mapping[str, str | Path]) -> dict[str, Any]:
    """Embed the raw text and hash of each config file used by a run."""
    snapshot: dict[str, Any] = {}
    for name, raw_path in config_paths.items():
        path = Path(raw_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if path.is_file():
            with path.open("r", encoding="utf-8") as handle:
                text = handle.read()
            snapshot[name] = {
                "path": str(raw_path),
                "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "text": text,
            }
        else:
            snapshot[name] = {"path": str(raw_path), "missing": True}
    return snapshot


def image_list_sha256(image_names: Iterable[str]) -> str:
    """Order-sensitive hash of an image list."""
    joined = "\n".join(str(name) for name in image_names)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def write_manifest(
    output_dir: str | Path,
    *,
    entry_point: str,
    args: Mapping[str, Any],
    config_paths: Mapping[str, str | Path],
    image_names: Sequence[str],
    checkpoints: Mapping[str, str | Path | None] | None = None,
    extra: Mapping[str, Any] | None = None,
    file_name: str = "manifest.json",
) -> Path:
    """Write ``manifest.json`` describing exactly what produced the files in ``output_dir``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_info = {
        name: {"path": str(path) if path else None, "sha256": file_sha256(path) if path else None}
        for name, path in (checkpoints or {}).items()
    }
    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "entry_point": entry_point,
        "argv": list(sys.argv),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in args.items()},
        "git": git_state(),
        "environment": environment_info(),
        "configs": config_snapshot(config_paths),
        "checkpoints": checkpoint_info,
        "images": {
            "count": len(image_names),
            "sha256": image_list_sha256(image_names),
            "names": list(image_names),
        },
        "extra": dict(extra or {}),
    }
    path = output_dir / file_name
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, default=str)
    return path


def load_image_list(path: str | Path, split: str | None = None) -> list[str]:
    """Read an image manifest written by scripts/make_manifests.py (or a plain text list).

    JSON manifests hold ``{"splits": {name: [file names]}}``; ``split`` selects one.
    Text files hold one file name per line.
    """
    path = Path(path)
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        splits = raw.get("splits", {})
        if split is None:
            if len(splits) != 1:
                raise ValueError(f"{path} has splits {sorted(splits)}; pass --split to choose one.")
            split = next(iter(splits))
        if split not in splits:
            raise KeyError(f"Split '{split}' not in {path} (available: {sorted(splits)})")
        return [str(name) for name in splits[split]]
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip() and not line.startswith("#")]


def resolve_image_paths(
    images_dir: str | Path,
    image_list: str | Path | None = None,
    split: str | None = None,
    num_images: int | None = None,
    extensions: Sequence[str] = (".jpg", ".jpeg", ".png"),
) -> list[Path]:
    """Images from a manifest when given, else the sorted prefix of ``images_dir`` (legacy)."""
    images_dir = Path(images_dir)
    if image_list is not None:
        names = load_image_list(image_list, split=split)
        paths = [images_dir / name for name in names]
        missing = [str(path) for path in paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} manifest images missing under {images_dir}, e.g. {missing[:3]}")
    else:
        paths = sorted(path for path in images_dir.iterdir() if path.suffix.lower() in extensions)
    if num_images is not None and num_images > 0:
        paths = paths[:num_images]
    return paths


__all__ = [
    "config_snapshot",
    "environment_info",
    "file_sha256",
    "git_state",
    "image_list_sha256",
    "load_image_list",
    "resolve_image_paths",
    "write_manifest",
]
