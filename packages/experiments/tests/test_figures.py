"""Tests of the mono-age figure and its log axes."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest
from conftest import MockRun

from experiments.figures import LOG_MARGIN, log_limits, log_ticks, mono_age_figure


def test_log_limits_hold_the_finite_positive_values() -> None:
    """The limits hold every finite positive value with a margin, ignoring NaN and non-positive values."""
    lo, hi = log_limits([np.array([2.0, np.nan, -1.0]), np.array([0.0, 50.0])])
    assert lo == pytest.approx(2.0 / LOG_MARGIN)
    assert hi == pytest.approx(50.0 * LOG_MARGIN)


def test_log_ticks() -> None:
    """Ticks fall at 1, 2 and 5 times powers of ten, with 3 and 7 added when the range is short."""
    assert log_ticks(0.3, 30.0) == pytest.approx([0.5, 1.0, 2.0, 5.0, 10.0, 20.0])
    assert log_ticks(14.0, 70.0) == pytest.approx([20.0, 30.0, 50.0, 70.0])


def test_mono_age_figure(mock_run: MockRun) -> None:
    """The figure has three log panels sharing the age axis, labeled, with a legend each."""
    fig = mono_age_figure(mock_run.analysis)
    top, middle, bottom = fig.axes
    assert [axes.get_yscale() for axes in fig.axes] == ["log"] * 3
    assert bottom.get_xlabel() == r"$\mathrm{age}\ (\mathrm{Gyr})$"
    assert middle.get_ylabel() == r"$h_Z\ (\mathrm{kpc})$"
    legends = [axes.get_legend() for axes in (top, middle, bottom)]
    assert all(legend is not None for legend in legends)
    middle_legend = middle.get_legend()
    assert middle_legend is not None
    assert "window 1 kpc" in [text.get_text() for text in middle_legend.get_texts()]
    plt.close(fig)
