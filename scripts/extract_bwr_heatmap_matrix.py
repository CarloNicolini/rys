#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy",
#     "pillow",
#     "opencv-python-headless",
#     "matplotlib",
#     "scikit-image",
# ]
# ///
"""
Load a diverging-colormap heatmap image, invert colors to z, resample to 64x64,
mask lower triangle with np.nan, save float matrix to NPZ.

The match reference is noise-free: RGB comes only from a matplotlib registered
colormap (default ``seismic``; use ``--cmap bwr`` if needed) and
``Normalize(vmin, vmax)`` on a dense z grid — never from the PNG.

Distances for choosing ``z`` are computed in **display RGB** by default (same
space as ``cmap(...)`` and typical PNGs). Optional ``--colorspace lab`` uses
CIELAB, which can mis-map warm tones that sit off the 1D colormap locus.

Sampling uses a **64×64 grid of cell medians** (inner crop per cell) so light
axes/margins and thin grid lines matter less than ``cv2.resize`` of the whole
figure.

Default scale: zmin (saturated blue) = -0.20, zmax (saturated red) = 0.05.

Run:
  uv run --with numpy --with pillow --with opencv-python-headless --with matplotlib \\
    --with scikit-image scripts/extract_bwr_heatmap_matrix.py

Or execute this file directly if your uv supports PEP 723 inline metadata.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from matplotlib import colormaps
from matplotlib.colors import Normalize
from PIL import Image
from skimage.color import rgb2lab


def noise_free_cmap_reference(
    cmap_name: str,
    vmin: float,
    vmax: float,
    n: int = 8192,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Dense (z, RGB) pairs for matching: RGB comes solely from the named matplotlib
    colormap and linear normalization on [vmin, vmax]. The input image is not used.
    """
    try:
        cmap = colormaps[cmap_name]
    except KeyError as e:
        raise ValueError(f"unknown matplotlib colormap: {cmap_name!r}") from e
    norm = Normalize(vmin=vmin, vmax=vmax)
    z_ref = np.linspace(vmin, vmax, n, dtype=np.float64)
    rgba = np.asarray(cmap(norm(z_ref)), dtype=np.float64)
    rgb_ref = rgba[:, :3]
    return z_ref, rgb_ref


def _rgb01_to_lab_flat(rgb: np.ndarray) -> np.ndarray:
    """rgb (N, 3) in [0, 1] -> LAB (N, 3)."""
    x = np.clip(rgb.astype(np.float64, copy=False), 0.0, 1.0)
    return rgb2lab(x.reshape(-1, 1, 3)).reshape(-1, 3)


def rgb_pairwise_d2(rgb_a: np.ndarray, rgb_b: np.ndarray) -> np.ndarray:
    """Squared Euclidean in RGB: rgb_a (N, 3), rgb_b (M, 3) -> (N, M)."""
    p2 = np.einsum("ij,ij->i", rgb_a, rgb_a, optimize=True)
    q2 = np.sum(rgb_b * rgb_b, axis=1)
    return p2[:, None] + q2[None, :] - 2.0 * (rgb_a @ rgb_b.T)


def lab_pairwise_d2(lab_a: np.ndarray, lab_b: np.ndarray) -> np.ndarray:
    """Squared Euclidean in LAB: lab_a (N, 3), lab_b (M, 3) -> (N, M)."""
    p2 = np.einsum("ij,ij->i", lab_a, lab_a, optimize=True)
    q2 = np.sum(lab_b * lab_b, axis=1)
    return p2[:, None] + q2[None, :] - 2.0 * (lab_a @ lab_b.T)


def rgb_to_z_idw(
    rgb: np.ndarray,
    z_lut: np.ndarray,
    rgb_lut: np.ndarray,
    *,
    space: str,
    eps: float = 1e-9,
) -> np.ndarray:
    """
    rgb: (..., 3) in [0, 1]. Returns z with same leading shape.

    Nearest samples on the matplotlib colormap locus in RGB or LAB, then inverse
    squared-distance z blend along three adjacent LUT indices.
    """
    flat = np.ascontiguousarray(rgb.reshape(-1, 3), dtype=np.float64)
    if space == "rgb":
        pix = flat
        lut = np.ascontiguousarray(rgb_lut, dtype=np.float64)
        d2_all = rgb_pairwise_d2(pix, lut)
    elif space == "lab":
        pix = _rgb01_to_lab_flat(flat)
        lut = _rgb01_to_lab_flat(rgb_lut)
        d2_all = lab_pairwise_d2(pix, lut)
    else:
        raise ValueError(f"unknown colorspace: {space!r}")

    i0 = np.argmin(d2_all, axis=1)
    m = lut.shape[0]

    im1 = np.clip(i0 - 1, 0, m - 1)
    ip1 = np.clip(i0 + 1, 0, m - 1)

    idx3 = np.stack([im1, i0, ip1], axis=1)
    lut3 = lut[idx3]
    d2 = np.sum((pix[:, None, :] - lut3) ** 2, axis=2)
    w = 1.0 / (d2 + eps)
    z3 = z_lut[idx3]
    z = np.sum(w * z3, axis=1) / np.sum(w, axis=1)
    return z.reshape(rgb.shape[:-1]).astype(np.float64)


def sample_grid_median(
    rgb: np.ndarray,
    size: int,
    *,
    inner: float = 0.55,
) -> np.ndarray:
    """
    Divide ``rgb`` (H, W, 3) into ``size``×``size`` cells; each output pixel is
    the **median** RGB over the central ``inner`` fraction of that cell (clipped
    to at least one pixel per axis).
    """
    if not 0.0 < inner <= 1.0:
        raise ValueError("inner must be in (0, 1]")
    h, w = rgb.shape[:2]
    out = np.empty((size, size, 3), dtype=np.float64)
    for i in range(size):
        y0 = int(round(i * h / size))
        y1 = int(round((i + 1) * h / size))
        y0, y1 = max(y0, 0), max(y1, y0 + 1)
        for j in range(size):
            x0 = int(round(j * w / size))
            x1 = int(round((j + 1) * w / size))
            x0, x1 = max(x0, 0), max(x1, x0 + 1)
            patch = rgb[y0:y1, x0:x1]
            ph, pw = patch.shape[:2]
            margin_y = int(round(0.5 * (1.0 - inner) * ph))
            margin_x = int(round(0.5 * (1.0 - inner) * pw))
            core = patch[margin_y : ph - margin_y, margin_x : pw - margin_x]
            if core.size == 0:
                core = patch
            out[i, j] = np.median(core.reshape(-1, 3), axis=0)
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description="Extract 64x64 z-values from a matplotlib diverging heatmap PNG."
    )
    p.add_argument(
        "--input",
        type=Path,
        default=Path.home() / "Desktop" / "image.png",
        help="Path to heatmap image",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path.home() / "Desktop" / "image_values.npz",
        help="Output .npz path",
    )
    p.add_argument(
        "--cmap",
        type=str,
        default="seismic",
        help="Matplotlib colormap name for the noise-free reference (e.g. seismic, bwr)",
    )
    p.add_argument(
        "--vmin",
        type=float,
        default=-0.20,
        help="z at colormap low end (e.g. dark blue for seismic)",
    )
    p.add_argument(
        "--vmax",
        type=float,
        default=0.05,
        help="z at colormap high end (e.g. dark red for seismic)",
    )
    p.add_argument("--size", type=int, default=64, help="Output matrix side length")
    p.add_argument(
        "--ref-samples",
        type=int,
        default=8192,
        help="Noise-free reference LUT length along z (matplotlib colormap only)",
    )
    p.add_argument(
        "--colorspace",
        choices=("rgb", "lab"),
        default="rgb",
        help="Distance space for matching to the colormap locus (rgb recommended)",
    )
    p.add_argument(
        "--sample",
        choices=("grid", "resize"),
        default="grid",
        help="grid = per-cell median; resize = cv2.resize whole image",
    )
    p.add_argument(
        "--cell-inner",
        type=float,
        default=0.55,
        help="With --sample grid, use central fraction of each cell for median",
    )
    p.add_argument(
        "--report-range",
        action="store_true",
        help="Print nanmin/nanmax of finite cells before writing NPZ",
    )
    args = p.parse_args()

    path = args.input.expanduser()
    if not path.is_file():
        raise SystemExit(f"Input not found: {path}")

    with Image.open(path) as im:
        rgb = np.asarray(im.convert("RGB"), dtype=np.float64) / 255.0
    size = args.size
    if args.sample == "grid":
        rgb_small = sample_grid_median(rgb, size, inner=args.cell_inner)
    else:
        rgb_small = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)

    z_ref, rgb_ref = noise_free_cmap_reference(
        args.cmap, args.vmin, args.vmax, n=args.ref_samples
    )
    mat = rgb_to_z_idw(rgb_small, z_ref, rgb_ref, space=args.colorspace)

    lower = np.tril_indices(size, k=-1)
    mat[lower] = np.nan

    if args.report_range:
        finite = mat[np.isfinite(mat)]
        if finite.size:
            print(
                f"Finite cells: min {finite.min():.6g} max {finite.max():.6g} (n={finite.size})"
            )

    out = args.output.expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, values=mat.astype(np.float64))

    print(
        f"Wrote {out} with array 'values' shape {mat.shape} dtype {mat.dtype} "
        f"(cmap={args.cmap!r}, vmin={args.vmin}, vmax={args.vmax})"
    )


if __name__ == "__main__":
    main()
