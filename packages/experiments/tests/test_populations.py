"""Tests of the mono-age measurement on a synthetic annulus whose populations are known.

Each 1 Gyr population has exponential heights of scale h(age), Gaussian v_R and v_z of
dispersions sigma_R(age) and sigma_z(age), and J_R spread around J(age). The vertical
potential is linear, Delta Phi = g |z|, so the tracer is an exponential of scale
sigma_z^2 / g. In the main sample the heights grow more slowly with age than that scale;
in the exact sample they follow it, so that the tracer is right at every age.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pytest

from experiments.coeval import THIN_LAYER_B, CoevalCurve, CoevalModel, coeval_curve
from experiments.cosmology import Cosmology
from experiments.heights import Z_GRID, sheet_hz
from experiments.jeans import TILT_R_MAX_KPC, TILT_R_MIN_KPC
from experiments.palicio import PalicioWindow
from experiments.populations import (
    AGE_BINS,
    AGE_RANGES,
    K_SCAN,
    R_MAX_KPC,
    R_MIN_KPC,
    Z_LIM_SCAN_KPC,
    AgeBin,
    AnnulusSample,
    Estimate,
    Method,
    MonoAge,
    VerticalModel,
    fit_eta,
    fit_grouped_log_slope,
    fit_log_slope,
    fit_power_law,
    fit_trend,
    measure_mono_age,
    range_exponents,
    surface_density,
)
from experiments.snapshot import Galaxy, Header, Particles, Stars
from experiments.stats import clipped_dispersion, half_width

G_LINEAR = 900.0  # (km/s)^2 / kpc
R_MID_KPC = 8.25
SIGMA_SURF = 50.0
N_PER_BIN = 4_000


def scale_height(age: float) -> float:
    """Return the scale height (kpc) of the population of *age* Gyr."""
    return 0.3 * (1.0 + age / 3.0)


def vertical_dispersion(age: float) -> float:
    """Return the vertical velocity dispersion (km/s) of the population of *age* Gyr."""
    return float(12.0 * (0.5 + age) ** 0.4)


def tracer_height(age: float) -> float:
    """Return the scale height (kpc) of the isothermal tracer of the population of *age* Gyr."""
    return vertical_dispersion(age) ** 2 / G_LINEAR


def vertical_action_scale(age: float) -> float:
    """Return the scale J_0 (kpc km/s) of the exponential distribution of the vertical actions at *age* Gyr."""
    return 2.0 * (0.5 + age)


@functools.cache
def linear_coeval() -> CoevalCurve:
    """Return the thin-layer population of the linear potential over the dispersions of the synthetic bins.

    The grid stops at 32 km/s, below the dispersion at which this shallow potential keeps more
    than EDGE_DENSITY of the midplane density at the edge of the heights (about 33 km/s).
    """
    return coeval_curve(CoevalModel(G_LINEAR * Z_GRID), np.geomspace(10.0, 32.0, 6), THIN_LAYER_B, 2.0)


def synthetic_sample(height: Callable[[float], float] = scale_height) -> AnnulusSample:
    """Return N_PER_BIN stars per 1 Gyr bin with heights of scale *height*, and the known dispersions and actions."""
    rng = np.random.default_rng(21)
    action_rng = np.random.default_rng(25)  # its own, so that the other columns do not depend on it
    keys = ("age", "z", "v_r", "v_z", "jr", "jz", "initial_mass", "r")
    parts: dict[str, list[np.ndarray]] = {key: [] for key in keys}
    for lo, hi in AGE_BINS:
        mid = 0.5 * (lo + hi)
        z = rng.exponential(height(mid), N_PER_BIN) * rng.choice([-1.0, 1.0], N_PER_BIN)
        jr = 10.0 * (0.5 + mid) * rng.lognormal(0.0, 0.3, N_PER_BIN)
        parts["age"].append(rng.uniform(lo, hi, N_PER_BIN))
        parts["z"].append(z)
        parts["v_r"].append(rng.normal(0.0, 1.5 * vertical_dispersion(mid), N_PER_BIN))
        parts["v_z"].append(rng.normal(0.0, vertical_dispersion(mid), N_PER_BIN))
        parts["jr"].append(np.where(np.abs(z) < 0.5, jr, np.nan))
        parts["jz"].append(action_rng.exponential(vertical_action_scale(mid), N_PER_BIN))
        parts["initial_mass"].append(np.ones(N_PER_BIN))
        parts["r"].append(rng.uniform(R_MIN_KPC, R_MAX_KPC, N_PER_BIN))
    return AnnulusSample(**{key: np.concatenate(value) for key, value in parts.items()})


def synthetic_outer(height: Callable[[float], float] = scale_height) -> AnnulusSample:
    """Return the populations of :func:`synthetic_sample` in the rings of the tilt term outside the annulus."""
    sample = synthetic_sample(height)
    rng = np.random.default_rng(22)
    inner = rng.uniform(size=sample.r.size) < 0.5
    r = np.where(
        inner,
        rng.uniform(TILT_R_MIN_KPC, R_MIN_KPC, sample.r.size),
        rng.uniform(R_MAX_KPC, TILT_R_MAX_KPC, sample.r.size),
    )
    return AnnulusSample(**{**sample.__dict__, "r": r, "jr": np.full(r.size, np.nan), "jz": np.full(r.size, np.nan)})


def measure(sample: AnnulusSample, outer: AnnulusSample | None = None) -> MonoAge:
    """Return the mono-age measurement of *sample* in the linear potential, with a snapshot's 5 percent deeper.

    Without *outer*, no star lies outside the annulus.
    """
    phi = G_LINEAR * Z_GRID
    model = VerticalModel(SIGMA_SURF, phi, phi + 0.05 * phi, R_MID_KPC, linear_coeval())
    outside = sample.select(np.zeros(sample.z.size, dtype=bool)) if outer is None else outer
    return measure_mono_age(sample, outside, model, Method(PalicioWindow(20.0, 1.0), 2.0, 20))


@pytest.fixture(name="measured", scope="module")
def fixture_measured() -> MonoAge:
    """Return the mono-age measurement of the synthetic annulus."""
    return measure(synthetic_sample())


def test_bins_recover_the_populations(measured: MonoAge) -> None:
    """Each bin recovers its h_Z within 3 bootstrap errors, its sigma_z, and the predictions of the sheet and tracer."""
    assert [(b.age_lo, b.age_hi) for b in measured.bins] == list(AGE_BINS)
    for b in measured.bins:
        mid = 0.5 * (b.age_lo + b.age_hi)
        kin = b.kinematics
        assert abs(b.heights.measured.value - scale_height(mid)) < 3.0 * b.heights.measured.err
        assert b.heights.window == pytest.approx(2.0 * b.heights.measured.value, rel=1e-6)
        assert kin.sigma_z.value == pytest.approx(vertical_dispersion(mid), rel=0.05)
        assert b.predictions.tracer == pytest.approx(kin.sigma_z.value**2 / G_LINEAR, rel=0.002)
        assert b.predictions.tracer_snapshot == pytest.approx(b.predictions.tracer / 1.05, rel=0.002)
        assert b.predictions.sheet == pytest.approx(sheet_hz(kin.sigma_z.value, SIGMA_SURF))
        assert b.slopes.measured.value == pytest.approx(kin.jr_median.value / b.heights.measured.value)
        assert b.slopes.tracer.value == pytest.approx(kin.jr_median.value / b.predictions.tracer)
        assert b.ratios.tracer.value == pytest.approx(b.predictions.tracer / b.heights.measured.value)
        assert b.ratios.sheet.value == pytest.approx(b.predictions.sheet / b.heights.measured.value)
        assert 0.0 < b.slopes.measured.err < b.slopes.measured.value
        assert 0.0 < b.ratios.tracer.err < b.ratios.tracer.value
        assert abs(b.half_mass.measured.value - np.log(2.0) * scale_height(mid)) < 3.0 * b.half_mass.measured.err
        assert 0.0 < b.half_mass.ratios.tracer.err < b.half_mass.ratios.tracer.value
        assert kin.n_mid < kin.n_jr < N_PER_BIN


def test_ranges_fit_the_age_exponents(measured: MonoAge) -> None:
    """Over each age range, eta is the fit of the heights against sigma_z, and 2 (beta_R - beta_z) follows."""
    assert [(r.lo, r.hi) for r in measured.ranges] == list(AGE_RANGES)
    for r in measured.ranges:
        exp = r.exponents
        chosen = [b for b in measured.bins if b.age_lo >= r.lo and b.age_hi <= r.hi]
        sigma_z = [b.kinematics.sigma_z for b in chosen]
        assert r.eta == fit_eta(sigma_z, [b.heights.measured for b in chosen])
        assert r.eta_half_mass == fit_eta(sigma_z, [b.half_mass.measured for b in chosen])
        assert r.sheet_exponent == pytest.approx(2.0 * (exp["sigma_r"].value - exp["sigma_z"].value))
    whole = measured.ranges[0].exponents
    assert whole["sigma_r"].value == pytest.approx(whole["sigma_z"].value, abs=0.05)


def test_bins_compare_the_thin_layer_population_and_fit_their_actions(measured: MonoAge) -> None:
    """Exponential actions give b = 1 and their J_0; the thin-layer ratios are the curve at sigma_z over the stars'."""
    for b in measured.bins:
        mid = 0.5 * (b.age_lo + b.age_hi)
        assert abs(b.coeval.b.value - 1.0) < max(3.0 * b.coeval.b.err, 0.05)
        assert b.coeval.j0.value == pytest.approx(vertical_action_scale(mid), rel=0.1)
        hz, half_mass = linear_coeval().at(b.kinematics.sigma_z.value)
        assert b.coeval.ratio.value == pytest.approx(hz / b.heights.measured.value)
        assert b.coeval.half_mass_ratio.value == pytest.approx(half_mass / b.half_mass.measured.value)
        assert 0.0 < b.coeval.ratio.err < b.coeval.ratio.value
        # The linear potential gives the isothermal tracer b = 4/3 (experiments.coeval).
        assert b.coeval.b_isothermal == pytest.approx(4.0 / 3.0, abs=0.02)
    for r in measured.ranges:
        assert np.isfinite(r.trends["coeval"].value) and np.isfinite(r.trends["half_mass_coeval"].value)


def test_trends_leave_out_bins_without_a_ratio(measured: MonoAge) -> None:
    """A bin with a NaN ratio drops out of the trend; with fewer than three bins left the trend is NaN."""
    bins = measured.bins
    gone = Estimate(float("nan"), float("nan"))

    def without(b: AgeBin) -> AgeBin:
        return replace(b, coeval=replace(b.coeval, ratio=gone))

    kept = range_exponents([without(b) if i < 2 else b for i, b in enumerate(bins)], 0.0, 10.0)
    expected = range_exponents(bins[2:], 0.0, 10.0)
    assert kept.trends["coeval"] == expected.trends["coeval"]
    assert kept.trends["tracer"] == range_exponents(bins, 0.0, 10.0).trends["tracer"]
    few = range_exponents([without(b) if i < 8 else b for i, b in enumerate(bins)], 0.0, 10.0)
    assert np.isnan(few.trends["coeval"].value)


def test_scans_cover_k_and_the_fixed_windows(measured: MonoAge) -> None:
    """The k scan adds Palicio's k_P to the scanned k; the window scan covers the fixed windows."""
    assert [row.parameter for row in measured.k_scan] == [*K_SCAN, pytest.approx(measured.k_palicio)]
    assert measured.k_palicio == pytest.approx(1.0 / measured.hz_all_fixed)
    assert [row.parameter for row in measured.window_scan] == list(Z_LIM_SCAN_KPC)
    main = next(row for row in measured.k_scan if row.parameter == 2.0)
    assert main.measured == pytest.approx(measured.ranges[0].exponents["a"].value)
    assert main.window_oldest == pytest.approx(2.0 * main.hz_oldest)


def test_clipped_dispersion_ignores_outliers() -> None:
    """Outliers beyond 4 MAD sigmas leave the dispersion of a Gaussian core unchanged."""
    v = np.random.default_rng(3).normal(0.0, 10.0, 100_000)
    assert clipped_dispersion(np.concatenate([v, np.full(500, 1.0e3)])) == pytest.approx(10.0, rel=0.01)


def test_half_width() -> None:
    """Half the 16th-84th percentile interval of a unit Gaussian is about 1; NaN draws are ignored, all NaN is NaN."""
    draws = np.random.default_rng(7).normal(0.0, 1.0, 100_000)
    assert half_width(draws) == pytest.approx(1.0, rel=0.02)
    assert half_width(np.append(draws, np.nan)) == half_width(draws)
    assert np.isnan(half_width(np.array([np.nan, np.nan])))


def test_estimate_without_a_value_has_no_error() -> None:
    """A value has the half width of its draws as its error; a NaN value has a NaN error, even with finite draws."""
    draws = np.array([0.5, 1.0, np.nan, 1.5])
    assert Estimate.of(1.0, draws) == Estimate(1.0, half_width(draws))
    assert np.isnan(Estimate.of(float("nan"), draws).err)


def test_trend_detects_a_tracer_with_the_wrong_age_dependence(measured: MonoAge) -> None:
    """The heights grow more slowly with age than the tracer of the linear potential: a trend of 0.19 over 0-10 Gyr.

    The ranges from 0 Gyr, whose youngest bins carry most of the lever arm, detect it at
    more than 5 errors.
    """
    for r in measured.ranges:
        assert set(r.trends) == {
            "sheet", "tracer", "tracer_snapshot", "jeans_sigma", "jeans",
            "half_mass_sheet", "half_mass_tracer", "half_mass_jeans", "coeval", "half_mass_coeval",
        }  # fmt: skip
        trend = r.trends["tracer"]
        if r.lo == 0.0:
            assert trend.value > 5.0 * trend.err


def test_trend_vanishes_when_the_tracer_is_right() -> None:
    """Heights that follow the tracer give ratios near 1 and trends within 3 errors of 0."""
    exact = measure(synthetic_sample(tracer_height))
    for b in exact.bins:
        assert b.ratios.tracer.value == pytest.approx(1.0, abs=4.0 * b.ratios.tracer.err)
        assert b.half_mass.ratios.tracer.value == pytest.approx(1.0, abs=4.0 * b.half_mass.ratios.tracer.err)
    for r in exact.ranges:
        assert abs(r.trends["tracer"].value) < 3.0 * r.trends["tracer"].err
        assert abs(r.trends["half_mass_tracer"].value) < 3.0 * r.trends["half_mass_tracer"].err


def test_jeans_heights_follow_the_tracer_of_an_isothermal_untilted_population() -> None:
    """With sigma_z constant in z and v_R independent of v_z, both Jeans heights equal the tracer's within errors."""
    exact = measure(synthetic_sample(tracer_height))
    for b in exact.bins:
        tolerance = 4.0 * b.ratios.tracer.err
        assert b.ratios.jeans_sigma.value == pytest.approx(b.ratios.tracer.value, abs=tolerance)
        assert b.ratios.jeans.value == pytest.approx(b.ratios.tracer.value, abs=tolerance)
        assert len(b.jeans.z) == len(b.jeans.sigma2) == len(b.jeans.tilt) > 0
    assert set(exact.ranges[0].trends) >= {"jeans_sigma", "jeans"}


def test_stars_outside_the_annulus_change_only_the_tilt_term() -> None:
    """The stars of the rings around the annulus enter only the tilt term, with resamplings of their own."""
    alone = measure(synthetic_sample(tracer_height))
    joined = measure(synthetic_sample(tracer_height), synthetic_outer(tracer_height))
    for a, b in zip(alone.bins, joined.bins, strict=True):
        assert (a.heights, a.kinematics, a.ratios.tracer, a.ratios.jeans_sigma) == (
            b.heights, b.kinematics, b.ratios.tracer, b.ratios.jeans_sigma
        )  # fmt: skip
        assert np.all(np.isnan(a.jeans.tilt)) and np.isfinite(b.jeans.tilt[0])
        assert b.ratios.jeans.value == pytest.approx(b.ratios.tracer.value, abs=4.0 * b.ratios.jeans.err)


def test_fit_trend_is_the_difference_of_the_exponents_with_one_set_of_weights() -> None:
    """With the same relative errors, the trend of y / y_pred is the exponent of y minus that of y_pred."""
    x = np.array([0.5, 1.5, 2.5, 4.0, 8.0])
    rng = np.random.default_rng(4)
    y, y_pred = 3.0 * x**0.7 * rng.lognormal(0.0, 0.05, x.size), 2.0 * x**0.4
    rel = np.array([0.02, 0.03, 0.05, 0.04, 0.08])
    difference = fit_power_law(x, y, rel * y).value - fit_power_law(x, y_pred, rel * y_pred).value
    assert fit_trend(x, y / y_pred, rel * y / y_pred).value == pytest.approx(difference)


def test_fit_log_slope_recovers_an_exponential_in_x() -> None:
    """An exponential 2 exp(0.5 x) has a slope of 0.5 in ln y against x, also at negative x, with no scatter."""
    x = np.array([-0.4, -0.1, 0.2, 0.5])
    y = 2.0 * np.exp(0.5 * x)
    trend = fit_log_slope(x, y, 0.05 * y)
    assert trend.value == pytest.approx(0.5) and trend.chi2_dof == pytest.approx(0.0, abs=1e-20)


def test_fit_grouped_log_slope_shares_the_slope_across_groups() -> None:
    """Two groups of different normalization share the slope 0.5 of ln y; one point per group leaves no dof."""
    x = np.array([-0.4, 0.0, 0.3, -0.2, 0.1, 0.4])
    group = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
    y = np.where(group == 1.0, 2.0, 5.0) * np.exp(0.5 * x)
    trend = fit_grouped_log_slope(x, y, 0.05 * y, group)
    assert trend.value == pytest.approx(0.5) and trend.chi2_dof == pytest.approx(0.0, abs=1e-20)
    alone = fit_grouped_log_slope(x[:2], y[:2], 0.05 * y[:2], np.array([1.0, 2.0]))
    assert np.isnan(alone.value) and np.isnan(alone.err)
    flat = fit_grouped_log_slope(np.repeat([0.1, 0.2], 3), y, 0.05 * y, group)
    assert np.isnan(flat.value)


def test_fit_grouped_log_slope_equals_the_fit_with_one_column_per_group() -> None:
    """The slope, its error and chi^2 equal those of weighted least squares with an intercept column per group."""
    rng = np.random.default_rng(8)
    x = rng.uniform(-0.5, 0.4, 30)
    group = rng.integers(0, 5, 30).astype(np.float64)
    y = np.exp(0.3 * x + 0.2 * group + rng.normal(0.0, 0.1, 30))
    y_err = rng.uniform(0.05, 0.2, 30) * y
    design = np.column_stack([*(np.asarray(group == g, dtype=np.float64) for g in np.unique(group)), x])
    w = (y / y_err) ** 2
    cov = np.linalg.inv(design.T @ (design * w[:, None]))
    coef = cov @ (design.T @ (w * np.log(y)))
    dof = x.size - design.shape[1]
    chi2_dof = float(np.sum(w * (np.log(y) - design @ coef) ** 2)) / dof
    trend = fit_grouped_log_slope(x, y, y_err, group)
    assert trend.value == pytest.approx(coef[-1], rel=1e-10) and trend.chi2_dof == pytest.approx(chi2_dof, rel=1e-10)
    assert trend.err == pytest.approx(np.sqrt(cov[-1, -1] * max(1.0, chi2_dof)), rel=1e-10)


def test_fit_trend_widens_its_error_by_the_scatter() -> None:
    """The error is the formal one while chi^2 per degree of freedom is below 1, and scales as its root above."""
    x = np.array([0.5, 1.5, 2.5, 4.0, 8.0])
    exact = fit_trend(x, 2.0 * x**0.3, 0.05 * 2.0 * x**0.3)
    assert exact.value == pytest.approx(0.3)
    assert exact.chi2_dof == pytest.approx(0.0, abs=1e-12)
    assert exact.err == pytest.approx(fit_power_law(x, 2.0 * x**0.3, 0.05 * 2.0 * x**0.3).err)
    y = 2.0 * x**0.3 * np.array([1.2, 0.8, 1.2, 0.8, 1.2])
    scattered = fit_trend(x, y, 0.05 * y)
    assert scattered.chi2_dof > 1.0
    assert scattered.err == pytest.approx(fit_power_law(x, y, 0.05 * y).err * np.sqrt(scattered.chi2_dof))


def test_fit_eta_recovers_the_slope_of_an_exact_power_law() -> None:
    """Heights h = 0.02 sigma^1.3 give eta = 1.3 with no scatter, whatever the errors of sigma."""
    sigma = np.array([10.0, 15.0, 25.0, 40.0, 70.0])
    height = 0.02 * sigma**1.3
    rel = np.array([0.02, 0.04, 0.03, 0.05, 0.08])
    fit = fit_eta([Estimate(s, r * s) for s, r in zip(sigma, rel)], [Estimate(h, 0.03 * h) for h in height])
    assert fit.value == pytest.approx(1.3) and fit.chi2_dof == pytest.approx(0.0, abs=1e-20)


def test_fit_eta_weights_by_the_errors_of_both_coordinates() -> None:
    """The fit is the weighted line of ln h on ln sigma with variances (dh/h)^2 + eta^2 (dsigma/sigma)^2 at its eta."""
    rng = np.random.default_rng(12)
    sigma = np.geomspace(10.0, 90.0, 8)
    height = 0.02 * sigma**1.2 * rng.lognormal(0.0, 0.05, 8)
    rel_s, rel_h = rng.uniform(0.02, 0.1, 8), rng.uniform(0.02, 0.06, 8)
    fit = fit_eta(
        [Estimate(s, r * s) for s, r in zip(sigma, rel_s)], [Estimate(h, r * h) for h, r in zip(height, rel_h)]
    )
    w = 1.0 / (rel_h**2 + fit.value**2 * rel_s**2)
    design = np.column_stack([np.ones(8), np.log(sigma)])
    cov = np.linalg.inv(design.T @ (design * w[:, None]))
    coef = cov @ (design.T @ (w * np.log(height)))
    chi2_dof = float(np.sum(w * (np.log(height) - design @ coef) ** 2)) / 6
    assert fit.value == pytest.approx(coef[1], rel=1e-8) and fit.chi2_dof == pytest.approx(chi2_dof, rel=1e-6)
    assert fit.err == pytest.approx(np.sqrt(cov[1, 1] * max(1.0, chi2_dof)), rel=1e-6)
    alone = fit_eta([Estimate(s, 1.0e-12 * s) for s in sigma], [Estimate(h, r * h) for h, r in zip(height, rel_h)])
    assert alone.value == pytest.approx(fit_power_law(sigma, height, rel_h * height).value, rel=1e-8)


def test_fit_power_law_recovers_an_exponent() -> None:
    """An exact power law gives its exponent with an error that scales with the errors of y."""
    x = np.array([0.5, 1.5, 2.5, 4.0, 8.0])
    fit = fit_power_law(x, 3.0 * x**0.7, 0.05 * 3.0 * x**0.7)
    assert fit.value == pytest.approx(0.7)
    assert isinstance(fit, Estimate)
    assert fit_power_law(x, 3.0 * x**0.7, 0.1 * 3.0 * x**0.7).err == pytest.approx(2.0 * fit.err)


def test_surface_density_counts_every_type_near_the_plane() -> None:
    """Sigma is the mass of every type with r_lo <= R < r_hi and |z| < 1.1 kpc, per pc^2."""
    pos = np.array([[8.0, 0.0, 0.0], [0.0, 8.5, 1.0], [8.0, 0.0, 2.0], [3.0, 0.0, 0.0]])
    stars = Stars(pos, np.zeros((4, 3)), np.full(4, 1.0e9), np.ones(4), np.ones(4), np.zeros(4))
    gas = Particles(pos[:1], np.zeros((1, 3)), np.array([2.0e9]))
    header = Header(1.0, 0.0, 67.77, (0.0,) * 6, Cosmology(0.6777, 0.307, 0.693))
    galaxy = Galaxy(header, stars, {0: gas})
    area_pc2 = np.pi * (9.0**2 - 7.5**2) * 1.0e6
    assert surface_density(galaxy, 7.5, 9.0) == pytest.approx(4.0e9 / area_pc2)
