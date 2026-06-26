"""Dataset helpers for Ng-style public EQ+MATH probes."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

MATH_FILES = {"16": "math_16.json", "120": "math_120.json"}
EQ_FILES = {"16": "eq_16.json", "140": "eq_140.json"}
BASE_URL = "https://raw.githubusercontent.com/dnhkng/RYS/main/datasets"


@dataclass(slots=True)
class DatasetRecord:
    name: str
    path: Path
    source: str
    sha256: str
    n_items: int


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()


def _download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=30) as response:
        data = response.read()
    dst.write_bytes(data)


def _resolve_file(filename: str, repo_root: Path) -> tuple[Path, str]:
    local = repo_root / "datasets" / filename
    if local.exists():
        return local, "local"
    cache = repo_root / ".cache" / "rys" / "datasets" / filename
    if not cache.exists():
        _download(f"{BASE_URL}/{filename}", cache)
    return cache, "remote"


def _load_dataset(filename: str, repo_root: Path, logical_name: str) -> tuple[dict, DatasetRecord]:
    path, source = _resolve_file(filename, repo_root)
    data = json.loads(path.read_text())
    rec = DatasetRecord(
        name=logical_name,
        path=path,
        source=source,
        sha256=_sha256(path),
        n_items=len(data),
    )
    return data, rec


def load_ng_datasets(
    *,
    repo_root: Path,
    math_set: str = "16",
    eq_set: str = "16",
) -> tuple[dict, dict, list[DatasetRecord]]:
    """Load canonical public MATH + EQ datasets from local-or-remote sources."""
    if math_set not in MATH_FILES:
        raise ValueError(f"Unsupported math set {math_set!r}; choose one of {sorted(MATH_FILES)}")
    if eq_set not in EQ_FILES:
        raise ValueError(f"Unsupported eq set {eq_set!r}; choose one of {sorted(EQ_FILES)}")
    math_data, math_rec = _load_dataset(MATH_FILES[math_set], repo_root, f"math_{math_set}")
    eq_data, eq_rec = _load_dataset(EQ_FILES[eq_set], repo_root, f"eq_{eq_set}")
    return math_data, eq_data, [math_rec, eq_rec]

