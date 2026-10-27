"""Shared matplotlib style and the writer of the figure set.

A plot is written to ``assets/figures`` as both a ``.png`` (for the documentation site)
and a ``.pdf`` (for the paper). A rendered image of a disk, which only the documentation
site shows, is written as a ``.jpg``, several times smaller than a PNG for a smooth
image. The new image is compared pixel for pixel with the existing one, and the files are
rewritten only when the content changed, so regenerating an unchanged figure produces no
spurious diff. Every text in a figure has one font size.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import AutoMinorLocator

FONT_SIZE = 9.0
DPI = 300
JPEG_OPTIONS = {"quality": 90, "optimize": True, "progressive": True}

# Figures are only written to files, never shown.
matplotlib.use("Agg")
matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "legend.fontsize": FONT_SIZE,
        "legend.title_fontsize": FONT_SIZE,
        "figure.titlesize": FONT_SIZE,
    }
)


def configure_axes(axes: Axes) -> None:
    """Draw inward major and minor ticks on all four sides of *axes*, with minor locators on linear axes."""
    if axes.get_xscale() == "linear":
        axes.xaxis.set_minor_locator(AutoMinorLocator())
    if axes.get_yscale() == "linear":
        axes.yaxis.set_minor_locator(AutoMinorLocator())
    axes.tick_params(which="minor", length=3, color="gray", direction="in")
    axes.tick_params(which="major", length=6, direction="in")
    axes.tick_params(top=True, right=True, which="both")


def apply_house_style(fig: Figure) -> None:
    """Style every axes of *fig*, make its legends frameless and fit the layout to the canvas."""
    for axes in fig.axes:
        configure_axes(axes)
        legend = axes.get_legend()
        if legend is not None:
            legend.set_frame_on(False)
    fig.tight_layout()


@dataclass(frozen=True)
class Saved:
    """The files of one figure, and whether they were written or left unchanged."""

    written: bool
    paths: tuple[Path, ...]

    def line(self) -> str:
        """Return the line that reports the outcome."""
        return f"  {'wrote' if self.written else 'unchanged'} {', '.join(str(path) for path in self.paths)}"


def _same_pixels(buffer: io.BytesIO, path: Path) -> bool:
    """Return whether the image in *buffer* has the pixels of the image file at *path*."""
    buffer.seek(0)
    new, old = plt.imread(buffer, format=path.suffix[1:]), plt.imread(path)
    return new.shape == old.shape and bool(np.array_equal(new, old))


def _save(fig: Figure, image: Path, pdf: Path | None, facecolor: str) -> Saved:
    """Write *fig* to *image* (PNG or JPEG) and to *pdf* if given, unless the image is unchanged; close it.

    The figure is saved at its exact size, without tight cropping, so that every figure
    of one width appears at one scale on the page.
    """
    jpeg = image.suffix == ".jpg"
    buffer = io.BytesIO()
    fig.savefig(
        buffer,
        format=image.suffix[1:],
        dpi=DPI,
        facecolor=facecolor,
        edgecolor="none",
        pil_kwargs=JPEG_OPTIONS if jpeg else None,
    )
    paths = (image,) if pdf is None else (image, pdf)
    unchanged = all(path.exists() for path in paths) and _same_pixels(buffer, image)
    if not unchanged:
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(buffer.getvalue())
        if pdf is not None:
            fig.savefig(pdf, format="pdf", dpi=DPI, facecolor=facecolor, edgecolor="none")
    plt.close(fig)
    return Saved(not unchanged, paths)


def save_figure(fig: Figure, name: str, figures_dir: Path) -> Saved:
    """Write the plot *fig* as ``<name>.png`` and ``<name>.pdf`` in *figures_dir* if it changed."""
    return _save(fig, figures_dir / f"{name}.png", figures_dir / f"{name}.pdf", "white")


def save_render(fig: Figure, name: str, figures_dir: Path) -> Saved:
    """Write the rendered image *fig* as ``<name>.jpg`` in *figures_dir* if it changed."""
    return _save(fig, figures_dir / f"{name}.jpg", None, "black")
