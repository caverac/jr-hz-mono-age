"""Tests of the figure style and of writing the figure set."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import AutoMinorLocator

from experiments._plotting import FONT_SIZE, Saved, apply_house_style, configure_axes, save_figure, save_render


def line_figure(slope: float) -> Figure:
    """Return a figure with one line of the given slope and a legend."""
    fig, axes = plt.subplots(figsize=(2.0, 2.0))
    axes.plot([0.0, 1.0], [0.0, slope], color="black", label="line")
    axes.legend()
    return fig


def test_save_figure_writes_only_changes(tmp_path: Path) -> None:
    """The PNG and PDF are written, kept when the figure is unchanged, and rewritten when it changes."""
    saved = save_figure(line_figure(1.0), "line", tmp_path)
    assert saved == Saved(True, (tmp_path / "line.png", tmp_path / "line.pdf"))
    assert all(path.is_file() for path in saved.paths)
    assert not save_figure(line_figure(1.0), "line", tmp_path).written
    assert save_figure(line_figure(2.0), "line", tmp_path).written
    (tmp_path / "line.pdf").unlink()
    assert save_figure(line_figure(2.0), "line", tmp_path).written
    assert save_figure(line_figure(2.0), "line", tmp_path).line() == (
        f"  unchanged {tmp_path / 'line.png'}, {tmp_path / 'line.pdf'}"
    )


def test_save_render_writes_a_jpeg(tmp_path: Path) -> None:
    """A rendered image is a JPEG alone, kept when unchanged."""
    saved = save_render(line_figure(1.0), "render", tmp_path)
    assert saved.paths == (tmp_path / "render.jpg",)
    assert (tmp_path / "render.jpg").read_bytes()[:3] == b"\xff\xd8\xff"
    assert not save_render(line_figure(1.0), "render", tmp_path).written
    assert not (tmp_path / "render.pdf").exists()


def test_house_style() -> None:
    """Linear axes get minor locators, log axes keep theirs, legends lose their frame, and text has one size."""
    fig, (linear, logarithmic) = plt.subplots(2, 1)
    logarithmic.set_yscale("log")
    logarithmic.set_xscale("log")
    linear.plot([1.0, 2.0], [1.0, 2.0], label="a")
    linear.legend()
    apply_house_style(fig)
    assert isinstance(linear.xaxis.get_minor_locator(), AutoMinorLocator)
    assert not isinstance(logarithmic.yaxis.get_minor_locator(), AutoMinorLocator)
    legend = linear.get_legend()
    assert legend is not None and not legend.get_frame_on()
    assert plt.rcParams["legend.fontsize"] == plt.rcParams["xtick.labelsize"] == FONT_SIZE
    configure_axes(linear)
    plt.close(fig)
