"""On-site data ingestion and quality validation.

Runs *inside* the hospital before any training. Produces two artefacts:

* ``local_report``: detailed, with file names, for the site's own data steward.
  It is written to disk at the site and never transmitted.
* ``shared_report``: aggregate counts only, with small-cell suppression (counts below
  ``k_min`` become "<k") and optional Laplace noise on the label histogram. This is what
  the coordinator receives, so it can spot broken sites and label skew without
  seeing a single image or file name.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from . import IMAGE_SHAPE, NUM_CLASSES
from .data import ArraySplit


@dataclass
class QualityResult:
    clean: ArraySplit
    local_report: dict
    shared_report: dict


def _dhash(arr: np.ndarray) -> int:
    """64-bit difference hash, used to flag near-duplicates (re-encoded exports)."""
    g = np.asarray(Image.fromarray(arr).convert("L").resize((9, 8), Image.Resampling.BILINEAR), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).reshape(-1)
    return int(sum(1 << i for i, b in enumerate(bits) if b))


def validate_site(
    site_dir: Path,
    expected_shape: tuple[int, int, int] = IMAGE_SHAPE,
    n_classes: int = NUM_CLASSES,
    k_min: int = 5,
    dp_epsilon: float | None = None,
    seed: int = 0,
    report_dir: Path | None = None,
    site_name: str | None = None,
) -> QualityResult:
    site_dir = Path(site_dir)
    issues: dict[str, list[str]] = {
        k: []
        for k in ("missing", "corrupt", "wrong_size", "wrong_mode", "bad_label",
                  "exact_duplicate", "near_duplicate", "low_contrast", "unlabelled")
    }
    with open(site_dir / "labels.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    labelled = {r["filename"] for r in rows}
    on_disk = {p.name for p in (site_dir / "images").glob("*")}
    issues["unlabelled"] = sorted(on_disk - labelled)

    xs, ys, hashes, dhashes = [], [], {}, {}
    pixel_means = []
    for r in rows:
        fname, raw_label = r["filename"], (r.get("label") or "").strip()
        # path traversal guard: labels.csv is input too
        if Path(fname).name != fname:
            issues["corrupt"].append(fname)
            continue
        p = site_dir / "images" / fname
        if not p.exists():
            issues["missing"].append(fname)
            continue
        try:
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im:
                mode = im.mode
                arr = np.asarray(im)
        except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
            issues["corrupt"].append(fname)
            continue
        if mode != "RGB" or arr.ndim != 3:
            issues["wrong_mode"].append(fname)
            continue
        if arr.shape != expected_shape:
            issues["wrong_size"].append(fname)
            continue
        if not raw_label.isdigit() or not 0 <= int(raw_label) < n_classes:
            issues["bad_label"].append(fname)
            continue
        h = hashlib.sha256(arr.tobytes()).hexdigest()
        if h in hashes:
            issues["exact_duplicate"].append(f"{fname}=={hashes[h]}")
            continue
        hashes[h] = fname
        dh = _dhash(arr)
        if dh in dhashes:
            issues["near_duplicate"].append(f"{fname}~{dhashes[dh]}")  # flagged, kept
        else:
            dhashes[dh] = fname
        if arr.std() < 2.0:
            issues["low_contrast"].append(fname)
            continue
        xs.append(arr)
        ys.append(int(raw_label))
        pixel_means.append(arr.reshape(-1, 3).mean(0))

    x = np.stack(xs) if xs else np.zeros((0, *expected_shape), np.uint8)
    y = np.array(ys, dtype=np.int64)
    counts = np.bincount(y, minlength=n_classes)
    excluded = sum(len(v) for k, v in issues.items() if k not in ("near_duplicate", "unlabelled"))

    local_report = {
        "site": site_name or site_dir.name,
        "n_listed": len(rows),
        "n_clean": int(len(y)),
        "n_excluded": excluded,
        "issues": issues,
        "label_counts": counts.tolist(),
    }
    report_dir = Path(report_dir) if report_dir else site_dir  # data mounts may be read-only
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "local_quality_report.json").write_text(json.dumps(local_report, indent=2))

    shared_counts = counts.astype(float)
    if dp_epsilon:
        # Laplace mechanism, sensitivity 1 (one patient changes one bin by 1)
        rng = np.random.default_rng(seed)
        shared_counts = np.maximum(0, shared_counts + rng.laplace(0, 1.0 / dp_epsilon, size=n_classes))
    shared_report = {
        "site": site_name or site_dir.name,
        "n_clean": _suppress(len(y), k_min),
        "n_excluded": _suppress(excluded, k_min, allow_zero=True),
        "issue_counts": {k: _suppress(len(v), k_min, allow_zero=True) for k, v in issues.items()},
        "label_histogram": [_suppress(int(round(c)), k_min, allow_zero=True) for c in shared_counts],
        "pixel_mean_rgb": np.round(np.mean(pixel_means, 0), 1).tolist() if pixel_means else None,
        "privacy": {"k_min": k_min, "dp_epsilon": dp_epsilon},
        "pass": excluded <= 0.05 * max(len(rows), 1) and len(y) >= 50,
    }
    return QualityResult(ArraySplit(x, y), local_report, shared_report)


def _suppress(n: int, k: int, allow_zero: bool = False) -> int | str:
    if n == 0 and allow_zero:
        return 0
    return n if n >= k else f"<{k}"


def summarise(reports: list[dict]) -> str:
    lines = [f"{'site':<12} {'clean':>6} {'excl':>5}  issues"]
    for r in reports:
        iss = ", ".join(f"{k}={v}" for k, v in r["issue_counts"].items() if v not in (0, "0"))
        lines.append(f"{r['site']:<12} {r['n_clean']!s:>6} {r['n_excluded']!s:>5}  {iss or '-'}")
    return "\n".join(lines)


def label_skew(reports: list[dict]) -> Counter:
    """Coordinator-side view of non-IID-ness from suppressed histograms."""
    c: Counter = Counter()
    for r in reports:
        for i, v in enumerate(r["label_histogram"]):
            if isinstance(v, int):
                c[i] += v
    return c
