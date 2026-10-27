"""Images of the stellar disks, rendered from their star particles and colored by age.

The data side (:func:`star_maps`, :func:`star_sample`) bins the star particles of a halo,
in its disk frame, into surface-density maps face-on (x, y) and edge-on (x, z), one per
age group of ``AGE_GROUPS_GYR``, within |x|, |y| < ``HALF_WIDTH_KPC`` and
|z| < ``HALF_HEIGHT_KPC``. It also draws a random sample of star particles for the
interactive view of the documentation, stored as int16 positions (pc) and uint16 ages
(Myr).

The image side (:func:`composite`) sets the brightness of a pixel by its stellar surface
density and its color by the mean age of its stars, from blue for the youngest to red for
the oldest (``AGE_COLORS``). The brightness is logarithmic over ``DYNAMIC_RANGE_DEX``
decades below the densest pixels, so that the center and the outskirts fit one image.
Faint pixels are smoothed more than bright ones, which removes the particle noise of the
outskirts without blurring the center. A blurred copy adds a soft glow, and an exposure
curve, 1 - exp(-x), saturates the brightest regions smoothly instead of clipping them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter

from experiments.snapshot import Stars

FloatArray = NDArray[np.float64]

HALF_WIDTH_KPC = 25.0
HALF_HEIGHT_KPC = 7.5
PIXELS = 400  # across the face-on map
AGE_GROUPS_GYR = ((0.0, 2.0), (2.0, 4.0), (4.0, 6.0), (6.0, 8.0), (8.0, 14.0))
GROUP_AGES_GYR = (1.0, 3.0, 5.0, 7.0, 11.0)  # the age that colors each group
AGE_COLORS = ("#3fa7ff", "#5ef2e8", "#ffe08a", "#ff8c42", "#ff3c5a")  # the colors at AGE_STOPS_GYR
AGE_STOPS_GYR = (0.0, *GROUP_AGES_GYR[1:])  # the age colormap starts at 0 Gyr
SAMPLE_SIZE = 80_000
SAMPLE_R_MAX_KPC, SAMPLE_Z_MAX_KPC = 30.0, 10.0
POSITION_UNIT_KPC = 1.0e-3
AGE_UNIT_GYR = 1.0e-3
DYNAMIC_RANGE_DEX = 3.0
WHITE_PERCENTILE = 99.9
GAMMA = 1.1
SMOOTH_PIXELS = (1.0, 3.0)  # the smoothing of the brightest and of the faintest pixels
GLOW_SIGMA_PIXELS = 6.0
GLOW = 0.35
EXPOSURE = 1.6
SCALE_BAR_KPC = 10.0
TEXT_COLOR = "white"


@dataclass(frozen=True)
class StarMaps:
    """Stellar surface densities (Msun/kpc^2) per age group, face-on (group, y, x) and edge-on (group, z, x)."""

    face_on: FloatArray
    edge_on: FloatArray
    half_width: float
    half_height: float


@dataclass(frozen=True)
class StarSample:
    """A random sample of star particles: positions in units of ``POSITION_UNIT_KPC``, ages of ``AGE_UNIT_GYR``."""

    positions: NDArray[np.int16]
    ages: NDArray[np.uint16]


def _edges(pixels: int) -> tuple[float, FloatArray, FloatArray]:
    """Return the pixel size (kpc) and the bin edges across the width and the height of the maps."""
    pixel = 2.0 * HALF_WIDTH_KPC / pixels
    n_z = int(round(2.0 * HALF_HEIGHT_KPC / pixel))
    return (
        pixel,
        np.linspace(-HALF_WIDTH_KPC, HALF_WIDTH_KPC, pixels + 1),
        np.linspace(-HALF_HEIGHT_KPC, HALF_HEIGHT_KPC, n_z + 1),
    )


def star_maps(stars: Stars, age: FloatArray, pixels: int = PIXELS) -> StarMaps:
    """Return the face-on and edge-on surface densities of *stars*, with ages *age* (Gyr), per age group."""
    pixel, width_edges, height_edges = _edges(pixels)
    half = np.array([HALF_WIDTH_KPC, HALF_WIDTH_KPC, HALF_HEIGHT_KPC])
    inside = np.all(np.abs(stars.pos) < half, axis=1)
    face, edge = [], []
    for lo, hi in AGE_GROUPS_GYR:
        chosen = inside & (age >= lo) & (age < hi)
        pos, weights = stars.pos[chosen], stars.mass[chosen] / pixel**2
        face.append(np.histogram2d(pos[:, 1], pos[:, 0], bins=(width_edges, width_edges), weights=weights)[0])
        edge.append(np.histogram2d(pos[:, 2], pos[:, 0], bins=(height_edges, width_edges), weights=weights)[0])
    return StarMaps(np.array(face), np.array(edge), HALF_WIDTH_KPC, HALF_HEIGHT_KPC)


def save_maps(maps: StarMaps, path: Path) -> Path:
    """Write *maps* to the compressed NumPy archive *path*, and return *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        face_on=maps.face_on.astype(np.float32),
        edge_on=maps.edge_on.astype(np.float32),
        half_width=maps.half_width,
        half_height=maps.half_height,
    )
    return path


def load_maps(path: Path) -> StarMaps:
    """Return the maps written by :func:`save_maps`."""
    with np.load(path) as archive:
        return StarMaps(
            np.asarray(archive["face_on"], dtype=np.float64),
            np.asarray(archive["edge_on"], dtype=np.float64),
            float(archive["half_width"]),
            float(archive["half_height"]),
        )


def star_sample(stars: Stars, age: FloatArray, size: int, rng: np.random.Generator) -> StarSample:
    """Return up to *size* star particles drawn at random within R < 30 kpc and |z| < 10 kpc."""
    inside = (stars.cylindrical_radius < SAMPLE_R_MAX_KPC) & (np.abs(stars.pos[:, 2]) < SAMPLE_Z_MAX_KPC) & (age >= 0.0)
    chosen = np.flatnonzero(inside)
    chosen = np.sort(rng.choice(chosen, size=min(size, chosen.size), replace=False))
    return StarSample(
        np.round(stars.pos[chosen] / POSITION_UNIT_KPC).astype(np.int16),
        np.round(age[chosen] / AGE_UNIT_GYR).astype(np.uint16),
    )


def write_sample(sample: StarSample, label: str, stem: Path) -> tuple[Path, Path]:
    """Write *sample* as ``<stem>.bin`` (little-endian int16 x, y, z per star, then uint16 ages) and its ``.json``."""
    binary, meta = stem.with_suffix(".bin"), stem.with_suffix(".json")
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_bytes(sample.positions.astype("<i2").tobytes() + sample.ages.astype("<u2").tobytes())
    meta.write_text(
        json.dumps(
            {
                "label": label,
                "count": int(sample.ages.size),
                "positionUnitKpc": POSITION_UNIT_KPC,
                "ageUnitGyr": AGE_UNIT_GYR,
                "ageGroupsGyr": [list(group) for group in AGE_GROUPS_GYR],
                "ageStopsGyr": list(AGE_STOPS_GYR),
                "ageColors": list(AGE_COLORS),
            },
            indent=2,
        )
        + "\n"
    )
    return binary, meta


def age_colormap() -> LinearSegmentedColormap:
    """Return the colormap of stellar age over [0, ``GROUP_AGES_GYR[-1]``], with ``AGE_COLORS`` at ``AGE_STOPS_GYR``."""
    stops = np.array(AGE_STOPS_GYR) / AGE_STOPS_GYR[-1]
    return LinearSegmentedColormap.from_list("age", list(zip(stops, AGE_COLORS, strict=True)))


def _brightness(density: FloatArray, dex: float) -> FloatArray:
    """Return log10 of *density* mapped to [0, 1] over *dex* decades below its brightest pixels."""
    top = float(np.log10(np.percentile(density[density > 0.0], WHITE_PERCENTILE)))
    return np.asarray(np.clip((np.log10(np.maximum(density, 1e-300)) - (top - dex)) / dex, 0.0, 1.0), dtype=np.float64)


def _adaptive_smooth(layers: FloatArray, dex: float) -> FloatArray:
    """Return *layers* smoothed by ``SMOOTH_PIXELS[0]`` pixels where bright, up to ``SMOOTH_PIXELS[1]`` where faint."""
    fine, coarse = (gaussian_filter(layers, sigma=(0.0, s, s)) for s in SMOOTH_PIXELS)
    weight = np.sqrt(_brightness(fine.sum(axis=0), dex))
    return np.asarray(weight * fine + (1.0 - weight) * coarse, dtype=np.float64)


def composite(layers: FloatArray, ages: tuple[float, ...], dex: float = DYNAMIC_RANGE_DEX) -> FloatArray:
    """Return the color image (y, x, rgb) of the surface densities *layers* (group, y, x) of groups of mean *ages*.

    A pixel is as bright as its total surface density and has the color of the
    mass-weighted mean age of its stars.
    """
    if not np.any(layers > 0.0):
        return np.zeros((*layers.shape[1:], 3))
    smooth = _adaptive_smooth(layers, dex)
    density = smooth.sum(axis=0)
    mean_age = np.tensordot(np.asarray(ages), smooth, axes=1) / np.maximum(density, 1e-300)
    color = age_colormap()(np.clip(mean_age / GROUP_AGES_GYR[-1], 0.0, 1.0))[..., :3]
    light = color * (_brightness(density, dex) ** GAMMA)[..., None]
    light = light + GLOW * gaussian_filter(light, sigma=(GLOW_SIGMA_PIXELS, GLOW_SIGMA_PIXELS, 0.0))
    return np.asarray(np.clip((1.0 - np.exp(-EXPOSURE * light)) / (1.0 - np.exp(-EXPOSURE)), 0.0, 1.0))


def _show(axes: Axes, image: FloatArray, half_width: float, half_height: float) -> None:
    """Draw *image* on *axes* over the given extent (kpc), without frame or ticks."""
    axes.imshow(image, origin="lower", extent=(-half_width, half_width, -half_height, half_height), aspect="equal")
    axes.set_axis_off()


def _scale_bar(axes: Axes, half_width: float, half_height: float) -> None:
    """Draw a ``SCALE_BAR_KPC`` bar in the lower left corner of *axes*."""
    x0, y0 = -0.9 * half_width, -half_height + 0.12 * min(half_width, half_height)
    axes.plot([x0, x0 + SCALE_BAR_KPC], [y0, y0], color="white", linewidth=1.2, solid_capstyle="butt")
    axes.text(x0 + 0.5 * SCALE_BAR_KPC, y0, f"{SCALE_BAR_KPC:g} kpc", ha="center", va="bottom", color=TEXT_COLOR)


def _age_key(axes: Axes) -> None:
    """Write the age groups (Gyr) in their colors along the bottom of *axes*."""
    for i, ((lo, hi), color) in enumerate(zip(AGE_GROUPS_GYR, AGE_COLORS, strict=True)):
        axes.text(
            0.98 - 0.13 * (len(AGE_GROUPS_GYR) - 1 - i),
            0.06,
            f"{lo:g}-{hi:g}",
            color=color,
            ha="right",
            va="bottom",
            transform=axes.transAxes,
        )
    axes.text(0.98, 0.94, "age (Gyr)", ha="right", va="top", transform=axes.transAxes, color=TEXT_COLOR)


def portrait_figure(maps: StarMaps, label: str) -> Figure:
    """Return the face-on and edge-on image of one disk, colored by age."""
    ratio = maps.half_height / maps.half_width
    fig = plt.figure(figsize=(3.5, 3.5 * (1.0 + ratio)), facecolor="black")
    face = fig.add_axes((0.0, ratio / (1.0 + ratio), 1.0, 1.0 / (1.0 + ratio)))
    edge = fig.add_axes((0.0, 0.0, 1.0, ratio / (1.0 + ratio)))
    _show(face, composite(maps.face_on, GROUP_AGES_GYR), maps.half_width, maps.half_width)
    _show(edge, composite(maps.edge_on, GROUP_AGES_GYR), maps.half_width, maps.half_height)
    face.text(0.03, 0.97, label, ha="left", va="top", transform=face.transAxes, color=TEXT_COLOR)
    _scale_bar(face, maps.half_width, maps.half_width)
    _age_key(edge)
    return fig
