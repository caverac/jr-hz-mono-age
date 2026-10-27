"""Tests of the mono-age measurement on a synthetic annulus whose populations are known.

Each 1 Gyr population has exponential heights of scale h(age), Gaussian v_R and v_z of
dispersions sigma_R(age) and sigma_z(age), and J_R spread around J(age). The vertical
potential is linear, Delta Phi = g |z|, so the tracer is an exponential of scale
sigma_z^2 / g.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.cosmology import Cosmology
from experiments.heights import Z_GRID, sheet_hz
from experiments.palicio import PalicioWindow
from experiments.populations import (
    AGE_BINS,
    AGE_RANGES,
    K_SCAN,
    Z_LIM_SCAN_KPC,
    AnnulusSample,
    Estimate,
    Method,
    MonoAge,
    VerticalModel,
    bootstrap,
    clipped_dispersion,
    fit_power_law,
    half_width,
    measure_mono_age,
    surface_density,
)
from experiments.snapshot import Galaxy, Header, Particles, Stars

G_LINEAR = 900.0  # (km/s)^2 / kpc
SIGMA_SURF = 50.0
N_PER_BIN = 4_000


def scale_height(age: float) -> float:
    """Return the scale height (kpc) of the population of *age* Gyr."""
    return 0.3 * (1.0 + age / 3.0)


def vertical_dispersion(age: float) -> float:
    """Return the vertical velocity dispersion (km/s) of the population of *age* Gyr."""
    return float(12.0 * (0.5 + age) ** 0.4)


def synthetic_sample() -> AnnulusSample:
    """Return N_PER_BIN stars per 1 Gyr bin with the known heights, dispersions and actions."""
    rng = np.random.default_rng(21)
    parts: dict[str, list[np.ndarray]] = {key: [] for key in ("age", "z", "v_r", "v_z", "jr", "initial_mass")}
    for lo, hi in AGE_BINS:
        mid = 0.5 * (lo + hi)
        z = rng.exponential(scale_height(mid), N_PER_BIN) * rng.choice([-1.0, 1.0], N_PER_BIN)
        jr = 10.0 * (0.5 + mid) * rng.lognormal(0.0, 0.3, N_PER_BIN)
        parts["age"].append(rng.uniform(lo, hi, N_PER_BIN))
        parts["z"].append(z)
        parts["v_r"].append(rng.normal(0.0, 1.5 * vertical_dispersion(mid), N_PER_BIN))
        parts["v_z"].append(rng.normal(0.0, vertical_dispersion(mid), N_PER_BIN))
        parts["jr"].append(np.where(np.abs(z) < 0.5, jr, np.nan))
        parts["initial_mass"].append(np.ones(N_PER_BIN))
    return AnnulusSample(**{key: np.concatenate(value) for key, value in parts.items()})


@pytest.fixture(name="measured", scope="module")
def fixture_measured() -> MonoAge:
    """Return the mono-age measurement of the synthetic annulus."""
    phi = G_LINEAR * Z_GRID
    model = VerticalModel(SIGMA_SURF, phi, phi + 0.05 * phi)
    return measure_mono_age(synthetic_sample(), model, Method(PalicioWindow(20.0, 1.0), 2.0, 10))


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
        assert b.slopes.tracer == pytest.approx(kin.jr_median.value / b.predictions.tracer)
        assert 0.0 < b.slopes.measured.err < b.slopes.measured.value
        assert kin.n_mid < kin.n_jr < N_PER_BIN


def test_ranges_fit_the_age_exponents(measured: MonoAge) -> None:
    """Over each age range, eta and 2 (beta_R - beta_z) follow from the exponents."""
    assert [(r.lo, r.hi) for r in measured.ranges] == list(AGE_RANGES)
    for r in measured.ranges:
        exp = r.exponents
        assert r.eta == pytest.approx(exp["hz"].value / exp["sigma_z"].value)
        assert r.sheet_exponent == pytest.approx(2.0 * (exp["sigma_r"].value - exp["sigma_z"].value))
    whole = measured.ranges[0].exponents
    assert whole["sigma_r"].value == pytest.approx(whole["sigma_z"].value, abs=0.05)


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


def test_bootstrap_and_half_width() -> None:
    """Bootstrap medians of a unit Gaussian scatter by about 1.25 / sqrt(n); NaN draws are ignored."""
    v = np.random.default_rng(6).normal(0.0, 1.0, 400)
    draws = bootstrap(np.random.default_rng(7), v.size, lambda i: float(np.median(v[i])), 500)
    assert draws.size == 500
    assert half_width(draws) == pytest.approx(1.2533 / np.sqrt(400), rel=0.2)
    assert half_width(np.append(draws, np.nan)) == half_width(draws)


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
