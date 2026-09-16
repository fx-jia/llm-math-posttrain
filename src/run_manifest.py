"""Reproducibility metadata saved next to training and evaluation artifacts."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


TRACKED_PACKAGES = (
    "torch",
    "transformers",
    "datasets",
    "peft",
    "trl",
    "accelerate",
    "math-verify",
    "openai",
)


def _package_versions() -> dict[str, str]:
    result = {}
    for package in TRACKED_PACKAGES:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "not-installed"
    return result


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _input_hashes(root: Path, values: dict) -> dict[str, str]:
    hashes = {}
    for key, value in values.items():
        if key.endswith("paths") and isinstance(value, (list, tuple)):
            candidates = value
        elif key.endswith("path") and isinstance(value, (str, Path)):
            candidates = [value]
        else:
            continue
        for candidate in candidates:
            path = Path(candidate)
            if not path.is_absolute():
                path = root / path
            if path.is_file():
                hashes[str(path)] = _file_sha256(path)
    return hashes


def write_manifest(path: Path, root: Path, arguments: dict, **extra) -> None:
    all_values = {**arguments, **extra}
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(root),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": _package_versions(),
        "arguments": arguments,
        "input_file_sha256": _input_hashes(root, all_values),
        **extra,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
