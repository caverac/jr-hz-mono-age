"""The figures of the paper, drawn from the records of the data commands.

Every figure is black, with line styles and filled or hollow markers in place of colors,
one font size, no title and units in parentheses, and panels that share an axis are
stacked (:mod:`experiments._plotting`):

1. ``eta``: the local slope of the tracer of each halo against sigma_z, with the eta fitted
   to its populations across their dispersions;
2. ``ratios``: the tracer, the Jeans equation with the tilt term and the sheet over the
   measured h_Z, against age, filled for the disk populations and hollow for those hotter
   than ``SPHEROID_SIGMA_Z``;
3. ``checks``: the trend of each check over 2-10 Gyr, halo by halo;
4. ``flare``: the whole-disk h_Z of each ring and the sum of tracers over it;
5. ``milky_way``: the tracer over the counts of each survey, in McMillan (2017) and across
   the admissible potentials, with the Jeans equation;
6. ``inversion``: the rho_0 at which every bin of a survey agrees, with the tracer and with the
   thin-layer population, against the measurement of McKee et al. (2015);
7. ``imig``: our exponential fits and the published heights of Imig et al. (2025) against
   sigma_z, cell by cell;
8. ``window``: the fixed over the self-consistent h_Z of injected profiles and of the Auriga
   populations.

Every ratio of two heights is drawn as its logarithm, with the gray band of agreement within
10 percent, ``|ln h_pred/h_Z| <= BAND``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FixedLocator, FormatStrFormatter, NullFormatter
from numpy.typing import NDArray

from experiments._plotting import apply_house_style
from experiments.analysis import HaloAnalysis
from experiments.figures import CAP_SIZE, ERROR_WIDTH, INK, LINE_WIDTH, MARKER_SIZE, log_ticks
from experiments.injection import Injection
from experiments.mw_imig_cells import ImigCell, ImigCells
from experiments.mw_pipeline import MilkyWay
from experiments.mw_potentials import RHO0
from experiments.populations import AgeBin, AgeRange, Trend
from experiments.robustness import CheckSet
from experiments.tracer_sum import TracerSumRecord

FloatArray = NDArray[np.float64]

WIDTH = 3.5  # inches, one column
SPHEROID_SIGMA_Z = 60.0  # km/s, the populations hotter than this are closer to a spheroid
BAND = 0.1  # the agreement within 10 percent, |ln h_pred/h_Z| <= BAND, drawn as a gray band
BAND_GRAY = "0.88"
GUIDE_GRAY = "0.55"
SURVEY_LABELS = {"lamost": "LAMOST", "apogee": "APOGEE (astroNN)", "apogee_distmass": "APOGEE (distmass)"}
SURVEY_MARKERS = {"lamost": "o", "apogee": "s", "apogee_distmass": "^"}


@dataclass(frozen=True)
class PaperRecords:
    """The records the figures read: the seven halos, their sums of tracers, the injection test and the Milky Way."""

    halos: list[HaloAnalysis]
    sums: list[TracerSumRecord]
    injection: Injection
    milky_way: MilkyWay


def _label(halo: HaloAnalysis) -> str:
    """Return the name of a halo, Au2 to Au27."""
    return f"Au{halo.snapshot.halo}"


def _range(halo: HaloAnalysis, lo: float, hi: float) -> AgeRange:
    """Return the range of ages [*lo*, *hi*] Gyr of *halo*."""
    return next(r for r in halo.mono_age.ranges if r.lo == lo and r.hi == hi)


def _band(axes: Axes) -> None:
    """Draw the line of equal heights, ln 1 = 0, and the gray band of agreement, ``|ln| <= BAND``."""
    axes.axhspan(-BAND, BAND, color=BAND_GRAY, linewidth=0.0, zorder=0)
    axes.axhline(0.0, color=INK, linewidth=LINE_WIDTH, zorder=1)


def _ln(ratios: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Return the logarithm of the ratios in the first column of *ratios* and its error.

    The second column holds the errors of the ratios, so that the error of the logarithm is,
    to first order, the relative error of the ratio. A group without populations, such as a
    halo with none hotter than ``SPHEROID_SIGMA_Z``, gives empty arrays.
    """
    pairs = np.reshape(ratios, (-1, 2))
    return np.log(pairs[:, 0]), pairs[:, 1] / pairs[:, 0]


def _log_axis(axes: Axes, which: str, lo: float, hi: float) -> None:
    """Set a log axis *which* ("x" or "y") over [*lo*, *hi*] with ticks at round values."""
    axis = axes.xaxis if which == "x" else axes.yaxis
    (axes.set_xscale if which == "x" else axes.set_yscale)("log")
    (axes.set_xlim if which == "x" else axes.set_ylim)(lo, hi)
    axis.set_major_locator(FixedLocator(log_ticks(lo, hi)))
    axis.set_major_formatter(FormatStrFormatter("%g"))
    axis.set_minor_formatter(NullFormatter())


def _style_with_legend_below(fig: Figure) -> None:
    """Apply the house style to a figure whose legend sits below its axes.

    The first layout pass shrinks the axes to make room for the legend, which moves up with
    them, so a second pass fits the layout to where the legend ends up.
    """
    apply_house_style(fig)
    fig.tight_layout()


def _errorbar(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    axes: Axes, x: FloatArray, y: FloatArray, yerr: FloatArray | None, marker: str, filled: bool, label: str | None
) -> None:
    """Draw black points with error bars, filled or hollow, without a joining line."""
    axes.errorbar(
        x,
        y,
        yerr=yerr,
        fmt=marker,
        color=INK,
        markerfacecolor=INK if filled else "white",
        markersize=MARKER_SIZE,
        capsize=CAP_SIZE,
        linewidth=ERROR_WIDTH,
        label=label,
    )


def eta_figure(records: PaperRecords) -> Figure:
    """Return the local slope of the tracer of each halo, and the eta fitted to its populations."""
    fig, axes = plt.subplots(figsize=(WIDTH, 2.8))
    for halo in records.halos:
        curve = halo.tracer_curve
        axes.plot(curve.sigma_z, curve.eta, "-", color=GUIDE_GRAY, linewidth=LINE_WIDTH)
        sigma = np.array([b.kinematics.sigma_z.value for b in halo.mono_age.bins if b.age_hi <= 10.0])
        eta = _range(halo, 0.0, 10.0).eta
        center = float(np.exp(np.mean(np.log(sigma))))
        axes.plot([sigma.min(), sigma.max()], [eta.value, eta.value], "-", color=INK, linewidth=ERROR_WIDTH)
        _errorbar(axes, np.array([center]), np.array([eta.value]), np.array([eta.err]), "o", True, None)
    for level, style in ((1.0, ":"), (2.0, "--")):
        axes.axhline(level, color=INK, linestyle=style, linewidth=LINE_WIDTH)
    _log_axis(axes, "x", 5.0, 150.0)
    axes.set_ylim(0.9, 2.1)
    axes.set_xlabel(r"$\sigma_z\ (\mathrm{km\,s^{-1}})$")
    axes.set_ylabel(r"$\eta$")
    apply_house_style(fig)
    return fig


def ratios_figure(records: PaperRecords) -> Figure:
    """Return the tracer, the Jeans equation with the tilt term and the sheet over the measured h_Z, against age.

    The populations hotter than ``SPHEROID_SIGMA_Z`` are hollow, the disk populations filled.
    """
    getters: tuple[tuple[str, Callable[[AgeBin], tuple[float, float]]], ...] = (
        (r"$\ln h_\mathrm{tracer}/h_Z$", lambda b: (b.ratios.tracer.value, b.ratios.tracer.err)),
        (r"$\ln h_\mathrm{Jeans}/h_Z$", lambda b: (b.ratios.jeans.value, b.ratios.jeans.err)),
        (r"$\ln h_\mathrm{sheet}/h_Z$", lambda b: (b.ratios.sheet.value, b.ratios.sheet.err)),
    )
    fig, panels = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(WIDTH, 6.2))
    bins = [b for halo in records.halos for b in halo.mono_age.bins]
    groups = (
        (True, r"$\sigma_z \leq 60\,\mathrm{km\,s^{-1}}$"),
        (False, r"$\sigma_z > 60\,\mathrm{km\,s^{-1}}$"),
    )
    for axes, (label, get) in zip(panels, getters, strict=True):
        _band(axes)
        for disk, legend in groups:
            chosen = [b for b in bins if (b.kinematics.sigma_z.value <= SPHEROID_SIGMA_Z) == disk]
            ln_ratio, err = _ln(np.array([get(b) for b in chosen]))
            age = np.array([b.age_median for b in chosen])
            _errorbar(axes, age, ln_ratio, err, "o", disk, legend if axes is panels[0] else None)
        axes.set_ylabel(label)
    panels[0].set_ylim(-0.9, 1.4)
    panels[0].legend(loc="upper left")
    panels[-1].set_xlabel(r"$\tau\ (\mathrm{Gyr})$")
    panels[-1].set_xlim(0.0, 10.0)
    apply_house_style(fig)
    return fig


CHECKS: tuple[tuple[str, str, bool, Callable[[HaloAnalysis], tuple[float, float]]], ...] = (
    ("tracer", "o", True, lambda h: _trend(_range(h, 2.0, 10.0).trends["tracer"])),
    ("disk stars", "o", False, lambda h: _check_trend(h.checks.disk, False)),
    ("bins of 0.25 Gyr", "s", True, lambda h: _check_trend(h.checks.fine, False)),
    (r"unclipped $\sigma_z$", "s", False, lambda h: _check_trend(h.checks.all_stars, True)),
    ("half-mass", "^", True, lambda h: _trend(_range(h, 2.0, 10.0).trends["half_mass_tracer"])),
    ("Jeans, tilt", "^", False, lambda h: _trend(_range(h, 2.0, 10.0).trends["jeans"])),
)


def _trend(trend: Trend) -> tuple[float, float]:
    """Return the value and the error of a trend."""
    return trend.value, trend.err


def _check_trend(check: CheckSet, unclipped: bool) -> tuple[float, float]:
    """Return the trend over 2-10 Gyr of the tracer of *check*, clipped or *unclipped*."""
    found = next(r for r in check.ranges if r.lo == 2.0 and r.hi == 10.0)
    return _trend(found.tracer_unclipped if unclipped else found.tracer)


def checks_figure(records: PaperRecords) -> Figure:
    """Return the trend of the tracer over 2-10 Gyr in every halo, in the main measurement and in each check."""
    fig, axes = plt.subplots(figsize=(WIDTH, 4.3))
    rows = np.arange(len(records.halos), dtype=np.float64)
    offsets = np.linspace(-0.3, 0.3, len(CHECKS))
    for (label, marker, filled, get), offset in zip(CHECKS, offsets, strict=True):
        values = np.array([get(halo) for halo in records.halos])
        axes.errorbar(
            values[:, 0], rows + offset, xerr=values[:, 1], fmt=marker, color=INK,
            markerfacecolor=INK if filled else "white", markersize=MARKER_SIZE, capsize=CAP_SIZE,
            linewidth=ERROR_WIDTH, label=label,
        )  # fmt: skip
    axes.axvline(0.0, color=INK, linewidth=LINE_WIDTH)
    axes.set_yticks(rows, [_label(h) for h in records.halos])
    axes.set_ylim(rows[-1] + 0.6, -0.6)
    axes.set_xlabel(r"$d\ln(h_\mathrm{pred}/h_Z)/d\ln\tau$, 2-10 Gyr")
    axes.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncols=2)
    _style_with_legend_below(fig)
    return fig


def flare_figure(records: PaperRecords) -> Figure:
    """Return the whole-disk h_Z of every ring of each halo, and the sum of tracers over it."""
    fig, (top, bottom) = plt.subplots(2, 1, sharex=True, figsize=(WIDTH, 4.4))
    for record in records.sums:
        rings = record.tracer_sum.rings
        r = np.array([0.5 * (ring.r_lo + ring.r_hi) for ring in rings])
        top.plot(r, [ring.measured.scaled.value for ring in rings], "-", color=INK, linewidth=ERROR_WIDTH)
        top.plot(r, [ring.mixture.scaled.value for ring in rings], ":", color=INK, linewidth=ERROR_WIDTH)
        ln_ratio, err = _ln(
            np.array([[ring.ratio_mixture.scaled.value, ring.ratio_mixture.scaled.err] for ring in rings])
        )
        _errorbar(bottom, r, ln_ratio, err, "o", True, None)
    top.plot([], [], "-", color=INK, linewidth=ERROR_WIDTH, label=r"measured $h_Z$")
    top.plot([], [], ":", color=INK, linewidth=ERROR_WIDTH, label="sum of tracers")
    top.legend(loc="upper left")
    top.set_ylabel(r"$h_Z\ (\mathrm{kpc})$")
    _band(bottom)
    bottom.set_ylim(-0.25, 0.25)
    bottom.set_ylabel(r"$\ln h_\mathrm{sum}/h_Z$")
    bottom.set_xlabel(r"$R\ (\mathrm{kpc})$")
    apply_house_style(fig)
    return fig


def milky_way_figure(records: PaperRecords) -> Figure:
    """Return the tracer and the Jeans equation over the counts of each survey, and the band of the admissible ones."""
    mw = records.milky_way
    admissible = [i for i, member in enumerate(mw.family) if member.admissible]
    fig, panels = plt.subplots(len(mw.surveys), 1, sharex=True, figsize=(WIDTH, 2.0 * len(mw.surveys) + 0.6))
    for axes, survey in zip(np.atleast_1d(panels), mw.surveys, strict=True):
        bins = [b for b in survey.bins if np.isfinite(b.ratios.tracer.value)]
        age = np.array([b.age_median for b in bins])
        family = np.log(np.array([[b.ratios.family[i] for i in admissible] for b in bins]))
        _band(axes)
        axes.fill_between(age, family.min(axis=1), family.max(axis=1), color=GUIDE_GRAY, alpha=0.5, linewidth=0.0)
        tracer, tracer_err = _ln(np.array([[b.ratios.tracer.value, b.ratios.tracer.err] for b in bins]))
        jeans, jeans_err = _ln(np.array([[b.ratios.jeans.value, b.ratios.jeans.err] for b in bins]))
        _errorbar(axes, age, tracer, tracer_err, "o", False, "tracer, McMillan (2017)")
        _errorbar(axes, age + 0.15, jeans, jeans_err, "s", True, r"Jeans, $\sigma_z(z)$")
        axes.set_ylim(-0.55, 0.4)
        axes.set_ylabel(r"$\ln h_\mathrm{pred}/h_Z$")
        axes.text(0.05, 0.04, SURVEY_LABELS[survey.name], transform=axes.transAxes)
    first = np.atleast_1d(panels)[0]
    first.fill_between([], [], [], color=GUIDE_GRAY, alpha=0.5, linewidth=0.0, label="tracer, admissible")
    first.legend(loc="upper left", ncols=1)
    np.atleast_1d(panels)[-1].set_xlabel(r"$\tau\ (\mathrm{Gyr})$")
    apply_house_style(fig)
    return fig


def inversion_figure(records: PaperRecords) -> Figure:
    """Return the rho_0 of each survey with the tracer and with the thin layer, against the measurement."""
    mw = records.milky_way
    fig, axes = plt.subplots(figsize=(WIDTH, 2.1))
    axes.axvspan(RHO0[0] - RHO0[1], RHO0[0] + RHO0[1], color=BAND_GRAY, linewidth=0.0, zorder=0)
    axes.axvline(RHO0[0], color=INK, linewidth=ERROR_WIDTH, linestyle="--", label="McKee et al. (2015)")
    unchanged = (1.0, 0.3, 1.0)
    mcmillan = next(
        m for m in mw.family if (m.params.thin_factor, m.params.thin_height_kpc, m.params.halo_factor) == unchanged
    )
    axes.axvline(mcmillan.rho0, color=INK, linewidth=ERROR_WIDTH, linestyle=":", label="McMillan (2017)")
    rows = np.arange(len(mw.inversions), dtype=np.float64)
    for inversions, offset, filled, label in (
        (mw.inversions, -0.12, True, "tracer"),
        (mw.inversions_coeval, 0.12, False, r"$b = 1/2$"),
    ):
        axes.errorbar(
            [inv.rho0.value for inv in inversions], rows + offset, xerr=[inv.rho0.err for inv in inversions],
            fmt="o", color=INK, markerfacecolor=INK if filled else "white", markersize=MARKER_SIZE, capsize=CAP_SIZE,
            linewidth=ERROR_WIDTH, label=label,
        )  # fmt: skip
    axes.set_yticks(rows, [SURVEY_LABELS[inv.survey] for inv in mw.inversions])
    axes.set_ylim(rows[-1] + 0.5, -0.5)
    axes.set_xlabel(r"$\rho_0\ (M_\odot\,\mathrm{pc^{-3}})$")
    axes.legend(loc="upper center", bbox_to_anchor=(0.5, -0.36), ncols=2)
    _style_with_legend_below(fig)
    return fig


def _measured(cells: ImigCells) -> list[ImigCell]:
    """Return the cells with enough matches and a sigma_z."""
    return [
        c
        for c in cells.cells
        if c.n_matched >= cells.min_matches
        and np.isfinite(c.kinematics.sigma_z.value)
        and np.isfinite(c.ours.exponential.value)
    ]


def _slope_lines(axes: Axes, sigma: FloatArray, height: FloatArray, low: float, high: float) -> None:
    """Draw the power laws of slope ``low``, ``high`` and 1 through the geometric mean of the points."""
    pivot_s, pivot_h = float(np.exp(np.mean(np.log(sigma)))), float(np.exp(np.mean(np.log(height))))
    line = np.array([sigma.min(), sigma.max()])
    for slope, style, label in (
        (low, "--", "tracer, admissible"),
        (high, "--", None),
        (1.0, ":", r"$\eta = 1$"),
    ):
        axes.plot(line, pivot_h * (line / pivot_s) ** slope, style, color=INK, linewidth=LINE_WIDTH, label=label)


def imig_figure(records: PaperRecords) -> Figure:
    """Return our exponential fits and the published heights of Imig et al. against the sigma_z of each cell."""
    mw = records.milky_way
    cells = _measured(mw.imig_cells)
    sigma = np.array([c.kinematics.sigma_z.value for c in cells])
    fig, axes = plt.subplots(figsize=(WIDTH, 3.0))
    ours = np.array([[c.ours.exponential.value, c.ours.exponential.err] for c in cells])
    theirs = np.array([[c.theirs.sun.value, c.theirs.sun.err] for c in cells])
    _errorbar(axes, sigma, ours[:, 0], ours[:, 1], "o", True, "our counts, exponential")
    _errorbar(axes, sigma, theirs[:, 0], theirs[:, 1], "o", False, "Imig et al. (2025)")
    growth = [m.growth.window for m in mw.family if m.admissible]
    _slope_lines(axes, sigma, ours[:, 0], min(growth), max(growth))
    _log_axis(axes, "x", 0.9 * float(sigma.min()), 1.1 * float(sigma.max()))
    heights = np.concatenate([ours[:, 0], theirs[:, 0]])
    _log_axis(axes, "y", 0.8 * float(np.nanmin(heights)), 1.25 * float(np.nanmax(heights)))
    axes.set_xlabel(r"$\sigma_z\ (\mathrm{km\,s^{-1}})$")
    axes.set_ylabel(r"$h_\mathrm{exp}\ (\mathrm{kpc})$")
    axes.legend(loc="upper left")
    apply_house_style(fig)
    return fig


def window_figure(records: PaperRecords) -> Figure:
    """Return the fixed over the self-consistent h_Z of the injected profiles and of the Auriga populations."""
    injection = records.injection
    fig, (top, bottom) = plt.subplots(2, 1, sharex=True, figsize=(WIDTH, 4.6))
    for family, style in (("exponential", "-"), ("sech2", "--"), ("mixture", ":")):
        points = sorted((p for p in injection.points if p.family == family), key=lambda p: p.reference)
        x = np.array([p.reference / injection.z_lim for p in points])
        label = r"$\mathrm{sech}^2$" if family == "sech2" else family
        top.plot(x, np.log([p.fixed.median for p in points]), style, color=INK, linewidth=LINE_WIDTH, label=label)
        top.fill_between(
            x, np.log([p.fixed.lo for p in points]), np.log([p.fixed.hi for p in points]),
            color=GUIDE_GRAY, alpha=0.5, linewidth=0.0,
        )  # fmt: skip
        bottom.plot(x, np.log([p.scaled.median for p in points]), style, color=INK, linewidth=LINE_WIDTH)
    for halo in records.halos:
        z_lim = halo.mono_age.method.window.z_lim
        bins = halo.mono_age.bins
        scaled = [b.heights.measured.value / z_lim for b in bins]
        top.plot(
            scaled, np.log([b.heights.fixed / b.heights.measured.value for b in bins]), "o", color=INK, markersize=2.0
        )
    _band(top)
    _band(bottom)
    top.set_xscale("log")
    top.set_ylabel(r"$\ln h_\mathrm{fixed}/h_Z$")
    top.legend(loc="upper left")
    bottom.set_ylim(-0.15, 0.15)
    bottom.set_ylabel(r"$\ln h_Z/h_\mathrm{ref}$")
    bottom.set_xlabel(r"$h_Z/Z_\mathrm{lim}$")
    bottom.xaxis.set_major_formatter(FormatStrFormatter("%g"))
    apply_house_style(fig)
    return fig


FIGURES: dict[str, Callable[[PaperRecords], Figure]] = {
    "paper-eta": eta_figure,
    "paper-ratios": ratios_figure,
    "paper-checks": checks_figure,
    "paper-flare": flare_figure,
    "paper-milky-way": milky_way_figure,
    "paper-inversion": inversion_figure,
    "paper-imig": imig_figure,
    "paper-window": window_figure,
}


def paper_figures(records: PaperRecords, names: Sequence[str] = tuple(FIGURES)) -> dict[str, Figure]:
    """Return the figures *names* of the paper drawn from *records*."""
    return {name: FIGURES[name](records) for name in names}
