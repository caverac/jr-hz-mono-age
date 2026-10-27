"""The figures of the analysis, drawn from the records that the data commands write.

:func:`mono_age_figure` shows the mono-age populations of one halo in three panels that
share the age axis:

- top: the measured slope a = J_R,med / h_Z with its bootstrap errors, against the
  isothermal tracer (dash-dot) and the self-gravitating sheet (dotted);
- middle: h_Z in the self-consistent window (filled) and in Palicio's fixed window
  (open), against the tracer and the sheet;
- bottom: the predicted over the measured h_Z, with the relative error of the measured
  h_Z.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FixedLocator, FormatStrFormatter, NullFormatter
from numpy.typing import NDArray

from experiments._plotting import apply_house_style
from experiments.analysis import HaloAnalysis
from experiments.populations import AgeBin

FloatArray = NDArray[np.float64]

AGE_LIM_GYR = (0.4, 12.0)
AGE_TICKS_GYR = (0.5, 1.0, 2.0, 4.0, 8.0)
LOG_MARGIN = 1.35
INK = "black"
LINE_WIDTH = 1.0
ERROR_WIDTH = 0.8
MARKER_SIZE = 4.0
CAP_SIZE = 2.0


def _column(bins: list[AgeBin], get: Callable[[AgeBin], float]) -> FloatArray:
    """Return one quantity of every bin as an array."""
    return np.array([get(b) for b in bins], dtype=np.float64)


def log_limits(arrays: Iterable[FloatArray]) -> tuple[float, float]:
    """Return limits of a log axis that hold every finite positive value of *arrays*, with a margin."""
    values = np.concatenate([np.asarray(a, dtype=np.float64).ravel() for a in arrays])
    values = values[np.isfinite(values) & (values > 0.0)]
    return float(values.min() / LOG_MARGIN), float(values.max() * LOG_MARGIN)


def log_ticks(lo: float, hi: float) -> list[float]:
    """Return ticks at 1, 2 and 5 times the powers of ten within [*lo*, *hi*], with 3 and 7 added if fewer than 3."""
    decades = range(int(np.floor(np.log10(lo))), int(np.ceil(np.log10(hi))) + 1)

    def within(mantissas: tuple[float, ...]) -> list[float]:
        return sorted(m * 10.0**d for d in decades for m in mantissas if lo <= m * 10.0**d <= hi)

    ticks = within((1.0, 2.0, 5.0))
    return ticks if len(ticks) >= 3 else within((1.0, 2.0, 3.0, 5.0, 7.0))


def _set_log_y(axes: Axes, arrays: Iterable[FloatArray]) -> None:
    """Set a log y axis that holds *arrays*, with labeled ticks at round values."""
    axes.set_yscale("log")
    lo, hi = log_limits(arrays)
    axes.set_ylim(lo, hi)
    axes.yaxis.set_major_locator(FixedLocator(log_ticks(lo, hi)))


def _line(axes: Axes, x: FloatArray, y: FloatArray, style: str, label: str) -> None:
    """Draw a black line of *style* (``":"`` for the sheet, ``"-."`` for the tracer)."""
    axes.plot(x, y, style, color=INK, linewidth=LINE_WIDTH, label=label)


def _points(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    axes: Axes, x: FloatArray, y: FloatArray, yerr: FloatArray | None, fmt: str, label: str
) -> None:
    """Draw black points with error bars; markers after a line style (``":s"``) are hollow and joined."""
    axes.errorbar(
        x,
        y,
        yerr=yerr,
        fmt=fmt,
        color=INK,
        markerfacecolor="white" if len(fmt) > 1 else INK,
        markersize=MARKER_SIZE,
        capsize=CAP_SIZE,
        linewidth=ERROR_WIDTH,
        label=label,
    )


def _slope_panel(axes: Axes, bins: list[AgeBin]) -> None:
    """Draw the measured a against the tracer and the sheet."""
    age = _column(bins, lambda b: b.age_median)
    sheet, tracer = _column(bins, lambda b: b.slopes.sheet.value), _column(bins, lambda b: b.slopes.tracer.value)
    measured = _column(bins, lambda b: b.slopes.measured.value)
    _line(axes, age, sheet, ":", "sheet")
    _line(axes, age, tracer, "-.", "tracer")
    _points(axes, age, measured, _column(bins, lambda b: b.slopes.measured.err), "o", "measured")
    _set_log_y(axes, (sheet, tracer, measured))
    axes.set_ylabel(r"$a\ (\mathrm{km\,s^{-1}})$")
    axes.legend(loc="best")


def _height_panel(axes: Axes, bins: list[AgeBin], z_lim: float) -> None:
    """Draw h_Z in the self-consistent and the fixed window against the tracer and the sheet."""
    age = _column(bins, lambda b: b.age_median)
    sheet, tracer = _column(bins, lambda b: b.predictions.sheet), _column(bins, lambda b: b.predictions.tracer)
    measured, fixed = _column(bins, lambda b: b.heights.measured.value), _column(bins, lambda b: b.heights.fixed)
    _line(axes, age, sheet, ":", "sheet")
    _line(axes, age, tracer, "-.", "tracer")
    _points(axes, age, measured, _column(bins, lambda b: b.heights.measured.err), "o", r"window $2h_Z$")
    axes.plot(
        age, fixed, "o", color=INK, markerfacecolor="white", markersize=MARKER_SIZE, label=f"window {z_lim:g} kpc"
    )
    _set_log_y(axes, (sheet, tracer, measured, fixed))
    axes.set_ylabel(r"$h_Z\ (\mathrm{kpc})$")
    axes.legend(loc="best")


def _ratio_panel(axes: Axes, bins: list[AgeBin]) -> None:
    """Draw the predicted over the measured h_Z for the tracer and the sheet."""
    age = _column(bins, lambda b: b.age_median)
    measured = _column(bins, lambda b: b.heights.measured.value)
    relative = _column(bins, lambda b: b.heights.measured.err) / measured
    ratios = []
    axes.axhline(1.0, color=INK, linewidth=LINE_WIDTH)
    for get, fmt, label in (
        (lambda b: b.predictions.sheet, ":s", "sheet"),
        (lambda b: b.predictions.tracer, "-.^", "tracer"),
    ):
        ratio = _column(bins, get) / measured
        ratios.append(ratio)
        _points(axes, age, ratio, ratio * relative, fmt, label)
    _set_log_y(axes, ratios)
    axes.set_ylabel(r"$h_{\mathrm{pred}}/h_Z$")
    axes.set_xlabel(r"$\mathrm{age}\ (\mathrm{Gyr})$")
    axes.legend(loc="best")


def mono_age_figure(analysis: HaloAnalysis) -> Figure:
    """Return the three-panel figure of the mono-age populations of one halo."""
    bins = analysis.mono_age.bins
    fig, (top, middle, bottom) = plt.subplots(3, 1, sharex=True, figsize=(3.5, 6.6))
    _slope_panel(top, bins)
    _height_panel(middle, bins, analysis.mono_age.method.window.z_lim)
    _ratio_panel(bottom, bins)
    top.set_xscale("log")
    top.set_xlim(*AGE_LIM_GYR)
    bottom.xaxis.set_major_locator(FixedLocator(AGE_TICKS_GYR))
    for axis in (bottom.xaxis, top.yaxis, middle.yaxis, bottom.yaxis):
        axis.set_major_formatter(FormatStrFormatter("%g"))
        axis.set_minor_formatter(NullFormatter())
    apply_house_style(fig)
    return fig
