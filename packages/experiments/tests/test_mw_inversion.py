"""Tests of the inversion of the tracer for the local vertical potential (experiments.mw_inversion)."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.mw_inversion import FamilyRatios, invert

TRUE_RHO0, TRUE_SIGMA = 0.10, 68.0


def family(slopes: list[tuple[float, float]], offsets: list[float] | None = None) -> FamilyRatios:
    """Return 36 members whose bins have ln r = c (ln rho_0 - ln TRUE_RHO0) + d (ln Sigma - ln TRUE_SIGMA) + offset."""
    rng = np.random.default_rng(5)
    rho0 = rng.uniform(0.08, 0.12, 36)
    sigma = rng.uniform(60.0, 80.0, 36)
    shifts = [0.0] * len(slopes) if offsets is None else offsets
    ratios = np.array(
        [
            np.exp(c * np.log(rho0 / TRUE_RHO0) + d * np.log(sigma / TRUE_SIGMA) + shift)
            for (c, d), shift in zip(slopes, shifts, strict=True)
        ]
    )
    return FamilyRatios(rho0, sigma, ratios, np.full(len(slopes), 0.03))


COLD_TO_HOT = [(-0.5, 0.0), (-0.4, -0.1), (-0.3, -0.2), (-0.2, -0.3), (-0.1, -0.4)]


def test_invert_recovers_the_potential_at_which_every_bin_agrees() -> None:
    """Exact bins that all reach 1 at one potential give it back, with no scatter about the family."""
    inversion = invert("test", family(COLD_TO_HOT))
    assert inversion.survey == "test" and inversion.n_bins == 5
    assert inversion.rho0.value == pytest.approx(TRUE_RHO0)
    assert inversion.sigma_11.value == pytest.approx(TRUE_SIGMA)
    assert inversion.chi2_dof == pytest.approx(0.0, abs=1e-12)
    assert inversion.model_rms == pytest.approx(0.0, abs=1e-12)
    assert all(b.value == pytest.approx(TRUE_RHO0) for b in inversion.bin_rho0)
    assert 0.0 < inversion.rho0.err < inversion.rho0.value and -1.0 < inversion.correlation < 1.0


def test_bins_that_disagree_widen_the_errors() -> None:
    """Offsets of 10 percent between bins give chi2 per dof above 1 and errors scaled by its root."""
    exact = invert("test", family(COLD_TO_HOT))
    scattered = invert("test", family(COLD_TO_HOT, [0.1, -0.1, 0.1, -0.1, 0.1]))
    assert scattered.chi2_dof > 1.0
    assert scattered.rho0.err > exact.rho0.err


def test_bins_without_ratios_are_left_out_and_too_few_give_nothing() -> None:
    """A bin with a NaN ratio or no error does not enter; fewer than three bins give NaN."""
    full = family(COLD_TO_HOT)
    ratios = full.ratios.copy()
    ratios[0, 3] = np.nan
    rel_err = full.rel_err.copy()
    rel_err[1] = np.nan
    partial = invert("test", FamilyRatios(full.rho0, full.sigma_11, ratios, rel_err))
    assert partial.n_bins == 3 and partial.rho0.value == pytest.approx(TRUE_RHO0)
    rel_err[2] = 0.0
    too_few = invert("test", FamilyRatios(full.rho0, full.sigma_11, ratios, rel_err))
    assert too_few.n_bins == 2 and np.isnan(too_few.rho0.value) and too_few.bin_rho0 == []
