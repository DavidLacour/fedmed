"""Dataset loading, non-IID partitioning and per-hospital export.

The dataset is DermaMNIST (MedMNIST v2), a 28x28 RGB version of HAM10000 dermatoscopic
images with 7 diagnostic classes. The partitioning helpers are dataset agnostic.
"""

from __future__ import annotations

import csv
import hashlib
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

DERMAMNIST_URL = "https://zenodo.org/records/10519652/files/dermamnist.npz?download=1"
DERMAMNIST_MD5 = "0744692d530f8e62ec473284d019b0c7"
DEFAULT_RAW = Path("data/raw/dermamnist.npz")


@dataclass
class ArraySplit:
    x: np.ndarray  # (N, H, W, C) uint8
    y: np.ndarray  # (N,) int64

    def __len__(self) -> int:
        return len(self.y)


def download_dermamnist(dest: Path = DEFAULT_RAW) -> Path:
    """Download DermaMNIST once and verify its MD5 checksum (integrity, not security)."""
    dest = Path(dest)
    if dest.exists() and _md5(dest) == DERMAMNIST_MD5:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    urllib.request.urlretrieve(DERMAMNIST_URL, tmp)  # noqa: S310 - fixed https URL
    if _md5(tmp) != DERMAMNIST_MD5:
        tmp.unlink()
        raise RuntimeError("Checksum mismatch for DermaMNIST download")
    tmp.rename(dest)
    return dest


def _md5(path: Path) -> str:
    h = hashlib.md5()  # noqa: S324 - checksum published by MedMNIST
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_npz(path: Path = DEFAULT_RAW) -> dict[str, ArraySplit]:
    d = np.load(path)
    return {
        split: ArraySplit(d[f"{split}_images"], d[f"{split}_labels"].reshape(-1).astype(np.int64))
        for split in ("train", "val", "test")
    }


def dirichlet_partition(
    labels: np.ndarray, n_sites: int, alpha: float, seed: int = 0, min_size: int = 30
) -> list[np.ndarray]:
    """Split sample indices across sites with label skew drawn from Dir(alpha).

    Small alpha (0.1) gives very different label mixes per hospital, large alpha (100)
    is close to IID. Resamples until each site has at least ``min_size`` samples.
    """
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    for _ in range(1000):
        site_idx: list[list[int]] = [[] for _ in range(n_sites)]
        for c in classes:
            idx = rng.permutation(np.flatnonzero(labels == c))
            props = rng.dirichlet(np.full(n_sites, alpha))
            cuts = (np.cumsum(props) * len(idx)).astype(int)[:-1]
            for s, part in enumerate(np.split(idx, cuts)):
                site_idx[s].extend(part.tolist())
        if min(len(s) for s in site_idx) >= min_size:
            return [np.sort(np.array(s, dtype=np.int64)) for s in site_idx]
    raise RuntimeError("Could not satisfy min_size; lower min_size or raise alpha")


def site_names(n_sites: int) -> list[str]:
    return [f"hospital-{chr(ord('a') + i)}" for i in range(n_sites)]


def to_tensor_dataset(split: ArraySplit) -> TensorDataset:
    x = torch.from_numpy(split.x).float().permute(0, 3, 1, 2) / 255.0
    x = (x - 0.5) / 0.5
    return TensorDataset(x, torch.from_numpy(split.y))


def make_loader(split: ArraySplit, batch_size: int, shuffle: bool, seed: int = 0) -> DataLoader:
    g = torch.Generator().manual_seed(seed)
    return DataLoader(to_tensor_dataset(split), batch_size=batch_size, shuffle=shuffle, generator=g)


def train_val_split(split: ArraySplit, val_frac: float, seed: int) -> tuple[ArraySplit, ArraySplit]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(split))
    n_val = max(1, int(round(len(split) * val_frac)))
    v, t = idx[:n_val], idx[n_val:]
    return ArraySplit(split.x[t], split.y[t]), ArraySplit(split.x[v], split.y[v])


# --------------------------------------------------------------------------- export
def export_sites(
    raw: Path,
    out_dir: Path,
    n_sites: int = 5,
    alpha: float = 0.5,
    seed: int = 0,
    inject_faults: bool = True,
) -> dict:
    """Materialise each hospital's private partition as PNG files + labels.csv.

    This mimics what a real site holds: a folder of image files and a label export
    from its PACS / EHR. ``inject_faults`` plants realistic defects in one site so the
    data-quality tool has something to find.
    """
    data = load_npz(raw)
    train = data["train"]
    parts = dirichlet_partition(train.y, n_sites, alpha, seed)
    out_dir = Path(out_dir)
    summary = {"alpha": alpha, "seed": seed, "sites": {}}
    for name, idx in zip(site_names(n_sites), parts):
        site = out_dir / name
        (site / "images").mkdir(parents=True, exist_ok=True)
        rows = []
        for k, i in enumerate(idx):
            fname = f"img_{k:05d}.png"
            Image.fromarray(train.x[i]).save(site / "images" / fname)
            rows.append((fname, int(train.y[i])))
        _write_labels(site / "labels.csv", rows)
        summary["sites"][name] = {
            "n": len(idx),
            "label_counts": np.bincount(train.y[idx], minlength=7).tolist(),
        }
    if inject_faults and n_sites >= 3:
        summary["injected_faults"] = inject_site_faults(out_dir / site_names(n_sites)[2], seed)
    # The official test split plays the role of a public benchmark held by the coordinator.
    bench = out_dir / "_benchmark"
    bench.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(bench / "test.npz", x=data["test"].x, y=data["test"].y)
    (out_dir / "partition.json").write_text(json.dumps(summary, indent=2))
    return summary


def _write_labels(path: Path, rows: list[tuple[str, int | str]]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "label"])
        w.writerows(rows)


def inject_site_faults(site: Path, seed: int = 0) -> dict:
    """Plant typical ingestion problems: corrupt files, wrong sizes, grayscale exports,
    duplicates, bad labels and a missing file."""
    rng = np.random.default_rng(seed + 99)
    labels_path = site / "labels.csv"
    with open(labels_path) as f:
        rows = list(csv.reader(f))[1:]
    files = [r[0] for r in rows]
    pick = rng.choice(len(files), size=14, replace=False)
    corrupt, resized, gray, dup_src, bad_label, missing = (
        pick[0:3], pick[3:5], pick[5:6], pick[6:10], pick[10:13], pick[13:14],
    )
    for i in corrupt:  # truncated PNG
        p = site / "images" / files[i]
        p.write_bytes(p.read_bytes()[:40])
    for i in resized:  # exported at the wrong resolution
        p = site / "images" / files[i]
        Image.open(p).resize((32, 32)).save(p)
    for i in gray:  # wrong colour mode
        p = site / "images" / files[i]
        Image.open(p).convert("L").save(p)
    extra = []
    for j, i in enumerate(dup_src):  # same image exported twice under a new name
        src = site / "images" / files[i]
        dst_name = f"dup_{j:03d}.png"
        (site / "images" / dst_name).write_bytes(src.read_bytes())
        extra.append([dst_name, rows[i][1]])
    for i in bad_label:
        rows[i][1] = ["9", "", "melanoma"][list(bad_label).index(i)]
    for i in missing:
        (site / "images" / files[i]).unlink()
    _write_labels(labels_path, [tuple(r) for r in rows + extra])
    return {
        "site": site.name,
        "corrupt": 3, "wrong_size": 2, "wrong_mode": 1, "duplicates": 4, "bad_label": 3, "missing": 1,
    }


def load_benchmark(out_dir: Path) -> ArraySplit:
    d = np.load(Path(out_dir) / "_benchmark" / "test.npz")
    return ArraySplit(d["x"], d["y"].astype(np.int64))
