"""Tests of the Milky Way test of the tracer, on the synthetic inputs of :mod:`mw_fixtures`."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import pyvo
from conftest import MOCK_SNAPSHOT, MockRun, invoke, run
from mw_fixtures import RC_PEAK, scale_height, write_world

from experiments import mw_counts
from experiments.commands import milky_way
from experiments.commands._report import write_json
from experiments.commands.auriga_analyze import result_name
from experiments.commands.auriga_tracer_sum import tracer_sum_name
from experiments.heights import Z_GRID, particle_hz, scaled_particle_hz, truncated_exp_scale
from experiments.injection import run_injection
from experiments.mw_analysis import BinSetting, SolarModel, Survey, measure_bin, solar_model
from experiments.mw_catalogs import (
    apogee_age_tracers,
    apogee_kinematics,
    lamost_age_tracers,
    lamost_kinematics,
    read_apogee,
    read_gaia,
    read_lamost,
    scattered_ages,
    with_distmass_ages,
)
from experiments.mw_checks import check_milky_way
from experiments.mw_counts import (
    APOGEE_BOX,
    BIAS_SCALES_KPC,
    GRADIENT_BOUND_PER_KPC,
    MW_FIT_RANGE,
    SLICE_EDGES_KPC,
    CountMoments,
    CountSample,
    Positions,
    apogee_count_sample,
    brighter_in_slice,
    count_hz,
    count_sample,
    exponential_slices,
    fit_luminosity_function,
    gaia_cylinder,
    horizontal_gradient,
    lamost_count_sample,
    population_counts,
    predicted_exponential_hz,
    predicted_hz,
    profile_slices,
    slice_fraction,
    slice_of,
)
from experiments.mw_dynamics import R0_KPC, V_SUN_KMS, Phase, Sky, cylindrical_actions, galactocentric, mcmillan17
from experiments.mw_imig import (
    NODE_R_KPC,
    NODE_WEIGHT,
    R_SUN_IMIG_KPC,
    cylinder_profile,
    cylinder_slices,
    pooled_slices,
    read_imig,
    split_normal,
)
from experiments.mw_pipeline import MilkyWay, analyze_milky_way, source_path
from experiments.mw_sources import GAIA_PATH, FileSource
from experiments.palicio import PalicioWindow
from experiments.paper_figures import FIGURES
from experiments.potential import circular_velocity
from experiments.records import load_milky_way, load_mw_checks
from experiments.tracer_sum import TracerSumRecord, measure_tracer_sum


@pytest.fixture(name="world_root", scope="module")
def fixture_world_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Return a root holding the synthetic inputs."""
    root = tmp_path_factory.mktemp("milky-way")
    write_world(root)
    return root


@pytest.fixture(name="result", scope="module")
def fixture_result(world_root: Path) -> MilkyWay:
    """Return the Milky Way test of the synthetic inputs, with small bootstraps."""
    return analyze_milky_way(world_root, n_bootstrap=3, n_draws=3)


def test_a_star_at_rest_next_to_the_sun_moves_with_the_sun() -> None:
    """A star 1 pc from the Sun, at rest with respect to it, sits at R_0 with the Sun's velocity."""
    one = np.ones(1)
    phase = galactocentric(Sky(0.0 * one, 90.0 * one, 1e-3 * one, 0.0 * one, 0.0 * one, 0.0 * one))
    assert phase.r[0] == pytest.approx(R0_KPC, abs=0.01)
    assert phase.v_t[0] == pytest.approx(V_SUN_KMS[1], abs=0.5)
    assert phase.v_z[0] == pytest.approx(V_SUN_KMS[2], abs=0.5)


def test_mcmillan17_and_its_solar_model() -> None:
    """The potential of McMillan (2017) has v_c = 233 km/s at 8.21 kpc, a rising Delta Phi, and J_R = 0 on a circle."""
    pot = mcmillan17()
    assert circular_velocity(pot, 8.21) == pytest.approx(233.1, abs=0.2)
    model = solar_model(pot)
    assert 60.0 < model.sigma_surf < 90.0
    assert model.phi[0] == 0.0 and np.all(np.diff(model.phi) > 0.0)
    v_c = circular_velocity(pot, R0_KPC)
    circular = Phase(np.array([R0_KPC]), np.zeros(1), np.zeros(1), np.array([v_c]), np.zeros(1))
    assert cylindrical_actions(pot, circular)[0] == pytest.approx(0.0, abs=0.1)


def test_catalogs_select_the_disk_of_the_solar_annulus(world_root: Path) -> None:
    """Both kinematic samples hold disk stars of the annulus; the age tracers carry their identifiers."""
    apogee = read_apogee(source_path(world_root, "astronn"), source_path(world_root, "allstar"))
    kin = apogee_kinematics(apogee)
    assert kin.age.size > 0 and np.all((kin.phase.r > 7.5) & (kin.phase.r < 9.0))
    tracers = apogee_age_tracers(apogee)
    assert np.all(tracers.source_id > 0) and tracers.in_disk.all()
    lamost = read_lamost(source_path(world_root, "lamost"))
    lamost_kin = lamost_kinematics(lamost, np.random.default_rng(1))
    assert lamost_kin.age.size > 0 and np.all(np.abs(lamost_kin.phase.z) < 1.5)
    assert np.all(lamost_kin.v_z_err > 0.0)
    assert lamost_kin.select(np.arange(3)).age.size == 3
    assert lamost_age_tracers(lamost).in_disk.all()
    distmass = with_distmass_ages(apogee, source_path(world_root, "distmass"))
    ages = distmass.num["age_lowess_correct"]
    assert np.isnan(ages[0])
    np.testing.assert_allclose(ages[1:], 1.1 * apogee.num["age_lowess_correct"][1:])
    assert np.nanmax(distmass.num["age_total_error"] / ages) < 0.4


def test_luminosity_function_finds_the_red_clump(world_root: Path) -> None:
    """The fit of the cylinder's luminosity function puts the red clump at its true M_W."""
    cylinder = gaia_cylinder(read_gaia(world_root / GAIA_PATH), MW_FIT_RANGE)
    fit = fit_luminosity_function(cylinder.m_w)
    assert fit.mu_rc == pytest.approx(RC_PEAK, abs=0.02)
    assert np.all(fit.density(np.linspace(*MW_FIT_RANGE, 5)) > 0.0)
    flat = fit.__class__(**{**fit.__dict__, "gamma": 0.0})
    assert np.all(np.isfinite(flat.density(np.array([-2.0]))))


def test_luminosity_function_fit_reports_a_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A sample without stars, and a fit that does not converge, are errors."""
    with pytest.raises(RuntimeError, match="no star"):
        fit_luminosity_function(np.full(3, np.nan))

    class Failed:
        """The result of a minimization that did not converge."""

        success, message, x = False, "stuck", np.zeros(8)

    monkeypatch.setattr(mw_counts, "minimize", lambda *args, **kwargs: Failed())
    with pytest.raises(RuntimeError, match="stuck"):
        fit_luminosity_function(np.array([-1.7, -1.8, -1.75]))


def test_count_samples_match_their_surveys(world_root: Path) -> None:
    """LAMOST matches by position and G inside the red clump; APOGEE by source_id inside the giant box."""
    gaia = read_gaia(world_root / GAIA_PATH)
    lamost_counts, clump = lamost_count_sample(gaia, lamost_age_tracers(read_lamost(source_path(world_root, "lamost"))))
    assert 0 < lamost_counts.tracer_age.size < lamost_counts.gaia_absz.size
    assert len(clump.fits) == 12
    apogee = read_apogee(source_path(world_root, "astronn"), source_path(world_root, "allstar"))
    apogee_counts = apogee_count_sample(gaia, apogee_age_tracers(apogee))
    cylinder = gaia_cylinder(gaia, APOGEE_BOX)
    assert apogee_counts.gaia_absz.size == cylinder.absz.size
    assert 0 < apogee_counts.tracer_age.size
    assert np.all(np.hypot(cylinder.x, 0.0) <= 1.5) and np.allclose(np.sort(apogee_counts.gaia_x), np.sort(cylinder.x))


def flat(absz: np.ndarray) -> Positions:
    """Return stars at heights *absz* without horizontal positions, x = 0."""
    return Positions(np.asarray(absz, dtype=np.float64), np.zeros(np.size(absz)))


def test_count_hz_recovers_the_scale_of_a_population() -> None:
    """Counts of one exponential population give its scale; a population thicker than the counts has none."""
    rng = np.random.default_rng(4)
    absz = rng.exponential(0.25, 200_000)
    absz = absz[absz < 1.2]
    sample = count_sample(flat(absz), flat(absz[:5000]), np.full(5000, 3.0), np.ones(5000, dtype=bool))
    assert np.all(slice_fraction(sample, 2.0, 4.0)[:8] == 1.0)
    hz, window = count_hz(sample, 2.0, 4.0, 2.0)
    assert hz == pytest.approx(0.25, rel=0.05) and window == pytest.approx(2.0 * hz, rel=1e-3)
    thick = rng.uniform(0.0, 1.2, 50_000)
    uniform = count_sample(flat(thick), flat(thick[:1000]), np.full(1000, 3.0), np.ones(1000, dtype=bool))
    assert np.all(np.isnan(count_hz(uniform, 2.0, 4.0, 2.0)))


CENTERS = 0.5 * (np.asarray(SLICE_EDGES_KPC)[1:] + np.asarray(SLICE_EDGES_KPC)[:-1])


@pytest.mark.parametrize("window", [1.0e-4, 0.05, 0.3, 1.0, 3.0])
@pytest.mark.parametrize("weighted", [False, True])
def test_count_moments_fit_as_the_scan_of_particle_hz(window: float, weighted: bool) -> None:
    """Running sums fit as particle_hz does, with each multiplicity a repeated star and the fraction interpolated.

    A weight of each star multiplies its fraction but not its count. The narrowest window has
    too few stars in its inner half, so both fits are NaN.
    """
    rng = np.random.default_rng(5)
    absz = rng.exponential(0.4, 25_000)
    absz = np.sort(absz[absz < 1.2])
    fraction = rng.uniform(0.1, 1.0, CENTERS.size)
    multiplicity = rng.integers(0, 3, absz.size).astype(np.float64)
    weight = rng.uniform(0.2, 3.0, absz.size) if weighted else np.ones(absz.size)
    repeats = multiplicity.astype(np.int64)
    star_weight = np.interp(absz, CENTERS, fraction) * weight
    expected = particle_hz(np.repeat(absz, repeats), np.repeat(star_weight, repeats), window)
    moments = CountMoments.of(absz, multiplicity, weight if weighted else None)
    assert moments.hz(fraction, window) == pytest.approx(expected, rel=1e-12, nan_ok=True)
    inside = absz <= 1.0
    mean = np.sum((multiplicity * star_weight * absz)[inside]) / np.sum((multiplicity * star_weight)[inside])
    assert moments.exponential(fraction, 1.0) == pytest.approx(truncated_exp_scale(float(mean), 1.0), rel=1e-12)


def test_count_moments_count_the_stars_of_each_slice_and_need_weight() -> None:
    """The stars of a slice carry their multiplicities, and a window without weight has no fit."""
    rng = np.random.default_rng(6)
    absz = np.sort(rng.uniform(0.0, 1.2, 5_000))
    multiplicity = rng.integers(0, 3, absz.size).astype(np.float64)
    moments = CountMoments.of(absz, multiplicity)
    expected = np.bincount(slice_of(absz), weights=multiplicity, minlength=CENTERS.size)
    assert moments.per_slice() == pytest.approx(expected)
    assert np.isnan(moments.hz(np.zeros(CENTERS.size), 0.5))
    assert np.isnan(moments.exponential(np.zeros(CENTERS.size), 0.5))


def test_count_moments_need_heights_in_ascending_order() -> None:
    """Running sums over unsorted heights would not hold the stars inside a window."""
    with pytest.raises(ValueError, match="ascending"):
        CountMoments.of(np.array([0.2, 0.1]), np.ones(2))


def test_slices_of_an_exponential() -> None:
    """The slices of an exponential hold its integral over each slice, and sum to its mass below 1.2 kpc."""
    slices = exponential_slices(0.3)
    assert slices.sum() == pytest.approx(0.3 * (1.0 - np.exp(-1.2 / 0.3)))
    assert profile_slices(np.exp(-Z_GRID / 0.3)) == pytest.approx(slices, rel=1e-4)


def test_a_prediction_goes_through_the_slices_of_the_counts() -> None:
    """A constant fraction gives the heights of the Gaia stars, whereas a thin exponential comes out thicker.

    The exponential fit, which matches the mean |z|, carries less of the bias of the slices
    than the windowed h_Z.
    """
    rng = np.random.default_rng(7)
    absz = rng.exponential(0.5, 300_000)
    absz = absz[absz < 1.2]
    sample = count_sample(flat(absz), flat(absz[:100]), np.ones(100), np.ones(100, dtype=bool))
    gaia = predicted_hz(sample, sample.moments.per_slice(), 2.0)
    assert gaia == pytest.approx(scaled_particle_hz(absz, np.ones_like(absz), 2.0), rel=1e-9)
    assert predicted_exponential_hz(sample, sample.moments.per_slice()) == pytest.approx(
        truncated_exp_scale(float(np.mean(absz)), 1.2), rel=1e-9
    )
    thin = exponential_slices(0.15)
    assert predicted_hz(sample, thin, 2.0)[0] > 1.1 * 0.15
    assert 0.15 < predicted_exponential_hz(sample, thin) < predicted_hz(sample, thin, 2.0)[0]
    assert predicted_hz(sample, exponential_slices(0.45), 2.0)[0] == pytest.approx(0.45, rel=0.02)
    assert predicted_exponential_hz(sample, exponential_slices(0.45)) == pytest.approx(0.45, rel=0.01)


def gradient_world(b: float, seed: int) -> tuple[CountSample, float]:
    """Return a sample whose population fraction is 0.3 exp(b x - |z|), and the h_Z of its exact weights.

    The Gaia stars are uniform in x and exponential in |z| with a scale of 0.4 kpc, so the
    population has a scale of about 0.29 kpc. The matches are drawn with a probability that
    falls toward the center most near the plane, as the APOGEE matches do.
    """
    rng = np.random.default_rng(seed)
    absz = rng.exponential(0.4, 600_000)
    absz = absz[absz < 1.2]
    x = rng.uniform(-1.5, 1.5, absz.size)
    observed = rng.random(absz.size) < 0.08 * (1.0 - 0.5 * (x / 1.5) * np.exp(-absz / 0.3))
    fraction = 0.3 * np.exp(b * x - absz)
    member = rng.random(absz.size) < fraction
    tracers = Positions(absz[observed], x[observed])
    sample = count_sample(Positions(absz, x), tracers, np.full(tracers.absz.size, 3.0), member[observed])
    order = np.argsort(absz)
    return sample, scaled_particle_hz(absz[order], fraction[order], 2.0)[0]


def test_the_gradient_of_a_population_is_recovered() -> None:
    """The fit returns the gradient of the population fraction, 0 without positions, and its bound beyond it."""
    sample, _ = gradient_world(0.6, 8)
    assert horizontal_gradient(sample, sample.tracer_in_disk) == pytest.approx(0.6, abs=0.05)
    n = sample.tracer_absz.size
    no_x = count_sample(flat(sample.gaia_absz), flat(sample.tracer_absz), np.ones(n), np.ones(n, dtype=bool))
    assert horizontal_gradient(no_x, sample.tracer_in_disk) == 0.0
    two = count_sample(
        flat(sample.gaia_absz),
        Positions(np.full(4, 0.05), np.array([-1.0, 1.0, -1.0, 1.0])),
        np.ones(4),
        np.ones(4, dtype=bool),
    )
    assert horizontal_gradient(two, np.array([False, True, False, True])) == GRADIENT_BOUND_PER_KPC
    assert horizontal_gradient(two, np.array([True, False, True, False])) == -GRADIENT_BOUND_PER_KPC


@pytest.mark.parametrize(("b", "shift"), [(-0.5, 1.06), (0.0, 1.0), (0.6, 0.925)])
def test_population_counts_correct_a_gradient(b: float, shift: float) -> None:
    """With matches toward the anticenter, the fraction of each slice misses a gradient; the correction does not.

    The truth weights every Gaia star by its exact fraction, without slices. Without the
    correction the population comes out thinner (b < 0) or thicker (b > 0); the correction
    moves it by *shift*, the same in every realization of the memberships, whose own scatter
    of about 4 percent the comparison with the truth allows, with the 2 percent of the slices.
    """
    sample, truth = gradient_world(b, 10)
    corrected = population_counts(sample, 2.0, 4.0)
    raw = sample.moments.fit(slice_fraction(sample, 2.0, 4.0), 2.0)[0]
    assert corrected.gradient == pytest.approx(b, abs=0.05)
    assert corrected.hz(2.0)[0] == pytest.approx(truth, rel=0.07)
    assert corrected.hz(2.0)[0] / raw == pytest.approx(shift, abs=0.015)
    weights = np.interp(sample.gaia_absz, CENTERS, corrected.level) * np.exp(corrected.gradient * sample.gaia_x)
    expected = np.bincount(slice_of(sample.gaia_absz), weights=weights, minlength=CENTERS.size)
    assert corrected.per_slice() == pytest.approx(expected, rel=1e-9)
    assert 0.0 < corrected.exponential_hz() < corrected.hz(2.0)[0] * 1.2


def test_count_sample_sorts_the_gaia_stars_and_ranks_the_catalog() -> None:
    """The heights are kept in ascending order, and the rank of each catalog row points at its star."""
    absz, x = np.array([0.5, 0.1, 0.9, 0.3]), np.array([1.0, 2.0, 3.0, 4.0])
    sample = count_sample(Positions(absz, x), flat(absz[:2]), np.full(2, 3.0), np.ones(2, dtype=bool))
    assert np.all(np.diff(sample.gaia_absz) >= 0.0)
    assert np.array_equal(sample.gaia_absz[sample.gaia_rank], absz)
    assert np.array_equal(sample.gaia_x[sample.gaia_rank], x)
    assert np.array_equal(sample.gaia_count, np.ones(4))


def test_a_resampled_count_sample_holds_the_stars_of_a_copy() -> None:
    """A draw enters the stars and matches that copying the drawn catalog rows would; a draw of it keeps its stars."""
    rng = np.random.default_rng(6)
    absz, tracer_absz, tracer_x = rng.exponential(0.3, 5000), rng.uniform(0.0, 1.2, 300), rng.uniform(-1.5, 1.5, 300)
    tracers = Positions(tracer_absz, tracer_x)
    sample = count_sample(flat(absz), tracers, rng.uniform(0.0, 10.0, 300), np.ones(300, dtype=bool))
    draw = sample.resampled(np.random.default_rng(7))
    copy = np.random.default_rng(7)
    rows, matches = copy.integers(0, absz.size, absz.size), copy.integers(0, 300, 300)
    assert np.array_equal(np.repeat(draw.gaia_absz, draw.gaia_count.astype(np.int64)), np.sort(absz[rows]))
    assert np.array_equal(draw.tracer_absz, tracer_absz[matches]) and np.array_equal(draw.tracer_x, tracer_x[matches])
    twice = draw.resampled(rng)
    assert twice.gaia_count.sum() == absz.size
    assert np.all(twice.gaia_count[draw.gaia_count == 0] == 0)


def test_imig_model_in_the_cylinder(world_root: Path) -> None:
    """Without flare or radial gradient their model is an exponential in every slice; HIGH rows are out.

    A flare mixes thinner and thicker exponentials, which raises the top slices, and the
    broken exponential weights each node by Sigma(R) / Sigma(R_sun) on its side of the break.
    """
    populations = read_imig(source_path(world_root, "imig"))
    assert [p.h.size for p in populations] == [2, 2, 2, 2]
    assert NODE_WEIGHT.sum() == pytest.approx(1.0)
    pops = populations[0]
    flat = replace(pops, a_flare=np.zeros(2), hr_in=np.full(2, 1.0e12), hr_out=np.full(2, 1.0e12))
    rows, h = np.array([0, 1]), np.array([0.3, 0.3])
    assert cylinder_slices(flat, rows, h) == pytest.approx(np.tile(exponential_slices(0.3), (2, 1)), rel=1e-9)
    assert cylinder_profile(flat, 0) == pytest.approx(np.exp(-Z_GRID / flat.h[0]), rel=1e-9)
    assert pooled_slices(flat, h, np.array([1.0, 2.0])) == pytest.approx(3.0 * exponential_slices(0.3), rel=1e-9)
    flared = replace(flat, a_flare=np.full(2, 0.1))
    assert cylinder_slices(flared, rows, h)[0, -1] > cylinder_slices(flat, rows, h)[0, -1]
    broken = replace(flat, hr_in=np.full(2, -5.0), hr_out=np.full(2, 2.0), r_break=np.full(2, R_SUN_IMIG_KPC))
    radial = np.exp(-(NODE_R_KPC - R_SUN_IMIG_KPC) / np.where(NODE_R_KPC <= R_SUN_IMIG_KPC, -5.0, 2.0))
    expected = (radial * NODE_WEIGHT).sum() * exponential_slices(0.3)
    assert cylinder_slices(broken, rows[:1], h[:1])[0] == pytest.approx(expected, rel=1e-9)
    with pytest.raises(ValueError, match="negative"):
        cylinder_slices(replace(flat, a_flare=np.full(2, 0.1)), rows, np.array([0.05, 0.05]))
    draws = split_normal(np.random.default_rng(2), np.full(1000, 1.0), np.tile([0.1, 0.3], (1000, 1)))
    assert np.all(draws > 0.0) and np.mean(draws > 1.0) == pytest.approx(0.5, abs=0.06)


def test_the_test_runs_on_both_surveys(result: MilkyWay) -> None:
    """Every survey measures every age bin and both ranges, and the Imig check measures four heights in four bins."""
    assert [s.name for s in result.surveys] == ["lamost", "apogee", "apogee_distmass"]
    for survey in result.surveys:
        assert len(survey.bins) == 5 and len(survey.ranges) == 2
        young = survey.bins[0]
        assert young.heights.measured.value == pytest.approx(scale_height(np.array(1.0))[()], rel=0.3)
        assert np.isfinite(young.ratios.tracer.value) and young.kinematics.n_kin > 0
        assert set(survey.ranges[0].trends) == {"sheet", "tracer", "jeans"}
        assert np.isfinite(young.ratios.jeans.value)
        assert len(young.ratios.family) == len(young.ratios.family_coeval) == len(result.family) == 36
        assert np.isfinite(young.ratios.coeval) and young.ratios.coeval > 0.0
        assert 0.9 < young.predictions.b_isothermal < 1.4
    for inversions in (result.inversions, result.inversions_coeval):
        assert [inv.survey for inv in inversions] == ["lamost", "apogee", "apogee_distmass"]
        assert all(inv.n_bins <= 5 and len(inv.bin_rho0) in (0, inv.n_bins) for inv in inversions)
    assert len(result.imig.bins) == 4
    assert set(result.imig.trends) == {"imig_mass", "imig_counts", "counts_astronn", "counts_distmass"}
    first = result.imig.bins[0]
    assert np.isfinite(first.heights.imig_counts.value) and first.kinematics.n_kin > 0
    assert first.ratios.imig_mass.value == pytest.approx(first.kinematics.tracer / first.heights.imig_mass.value)
    assert len(result.rc_peak) == 12 and result.r0_kpc == R0_KPC and result.slice_edges_kpc == SLICE_EDGES_KPC
    for survey in result.surveys:
        assert survey.bias.scales == list(BIAS_SCALES_KPC) and survey.bias.ratios[0] > 1.0
        assert all(np.isfinite(b.heights.gradient) for b in survey.bins if b.heights.n_matched > 0)
    assert len(result.estimators) == 6
    assert all(e.windowed > e.exponential >= e.exponential_whole > 0.0 for e in result.estimators)
    assert len(result.imig_cells.cells) == 8 and result.imig_cells.min_matches == 100
    assert len(result.imig_cells_lamost.cells) == 8
    boxed = [t.cells.n_matches for t in result.imig_cells_targeting]
    assert len(boxed) == 2 and all(0 < n < result.imig_cells.n_matches for n in boxed)


def test_a_bin_without_matches_has_no_median_age(world_root: Path) -> None:
    """An age range that no matched star falls in has no median age and drops out of the ranges."""
    gaia = read_gaia(world_root / GAIA_PATH)
    apogee = read_apogee(source_path(world_root, "astronn"), source_path(world_root, "allstar"))
    kin = apogee_kinematics(apogee)
    survey = Survey("apogee", apogee_count_sample(gaia, apogee_age_tracers(apogee)), kin, np.full(kin.age.size, 10.0))
    model = SolarModel(np.asarray(900.0 * Z_GRID), 50.0)
    empty = measure_bin(survey, (20.0, 30.0), BinSetting(model, np.random.default_rng(3), 2))
    assert np.isnan(empty.age_median) and empty.heights.n_matched == 0


def test_scattered_ages_double_the_variance_of_the_errors() -> None:
    """Lognormal draws with err / age as the width of ln age; ages without a good error stay as they are."""
    rng = np.random.default_rng(6)
    age, err = np.full(20000, 5.0), np.full(20000, 1.0)
    drawn = scattered_ages(age, err, rng)
    assert np.all(drawn > 0.0)
    assert np.std(np.log(drawn / age)) == pytest.approx(0.2, rel=0.03)
    kept = scattered_ages(np.array([np.nan, 4.0, 4.0]), np.array([1.0, np.nan, 0.0]), rng)
    assert np.isnan(kept[0]) and kept[1] == 4.0 and kept[2] == 4.0


def test_matches_split_at_the_median_g_of_their_slice() -> None:
    """In each slice of |z| the bright half lies below the median G of its slice; a split keeps every Gaia star."""
    absz = np.array([0.05, 0.05, 0.05, 0.05, 0.55, 0.55])
    g = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
    sample = count_sample(flat(absz), flat(absz), np.full(6, 3.0), np.ones(6, dtype=bool), g)
    bright = brighter_in_slice(sample)
    assert bright.tolist() == [True, True, False, False, True, False]
    half = sample.with_matches(bright)
    assert half.tracer_g.tolist() == [10.0, 11.0, 14.0] and half.gaia_absz.size == sample.gaia_absz.size
    assert np.all(np.isnan(count_sample(flat(absz), flat(absz), np.ones(6), np.ones(6, dtype=bool)).tracer_g))


def test_records_read_back(result: MilkyWay, tmp_path: Path) -> None:
    """The record of the test reads back into an equal result."""
    written = write_json(result, tmp_path / "milky-way.json")
    again = write_json(load_milky_way(written), tmp_path / "again.json")
    assert again.read_text() == written.read_text()


def test_checks_remeasure_every_survey(world_root: Path, tmp_path: Path) -> None:
    """Each realization remeasures the three surveys, and each survey is split at the median G of its tracers."""
    result = check_milky_way(world_root, n_realizations=2, n_bootstrap=3)
    written = write_json(result, tmp_path / "checks.json")
    assert write_json(load_mw_checks(written), tmp_path / "again.json").read_text() == written.read_text()
    assert result.n_bootstrap == 3 and len(result.age_errors) == 2
    names = ["lamost", "apogee", "apogee_distmass"]
    assert all([s.name for s in surveys] == names for surveys in result.age_errors)
    assert [m.survey for m in result.magnitude] == names
    for split in result.magnitude:
        assert np.isfinite(split.g_split)
        assert len(split.bright.bins) == len(split.faint.bins) == 5
        assert abs(split.bright.n_matched - split.faint.n_matched) <= 0.1 * (
            split.bright.n_matched + split.faint.n_matched
        )


def test_milky_way_commands(world_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``check`` fails on synthetic files, whose checksums differ; ``analyze`` writes a record; ``fetch`` fetches.

    ``analyze`` reports each of its nine stages on stderr as it ends, and its report on stdout.
    """
    result = run("milky-way", "check", "--root", str(world_root))
    assert result.exit_code == 1 and "FAILED" in result.output
    analyzed = run("milky-way", "analyze", "--root", str(world_root), "--n-bootstrap", "2", "--out-dir", str(tmp_path))
    assert analyzed.exit_code == 0 and analyzed.stdout.startswith("Milky Way at R_0")
    stages = analyzed.stderr.splitlines()
    assert len(stages) == 9 and all(line.startswith("[") and " min] " in line for line in stages)
    assert "] lamost: 5 age bins, tracer over h_Z" in stages[1] and "] LAMOST cells: " in stages[-1]
    record = json.loads((tmp_path / "milky-way.json").read_text())
    assert [s["name"] for s in record["surveys"]] == ["lamost", "apogee", "apogee_distmass"]
    missing = run("milky-way", "analyze", "--root", str(tmp_path / "nothing"))
    assert missing.exit_code == 1 and "experiments milky-way fetch" in missing.output
    checked = run(
        "milky-way", "checks", "--root", str(world_root), "--realizations", "1", "--n-bootstrap", "2",
        "--out-dir", str(tmp_path),
    )  # fmt: skip
    assert checked.exit_code == 0 and checked.stdout.startswith("ages drawn again")
    assert len(checked.stderr.splitlines()) == 2
    assert set(json.loads((tmp_path / "checks.json").read_text())) == {"n_bootstrap", "age_errors", "magnitude"}
    missing = run("milky-way", "checks", "--root", str(tmp_path / "nothing"))
    assert missing.exit_code == 1 and "experiments milky-way fetch" in missing.output

    fetched: list[str] = []

    def fake_fetch_file(source: FileSource, root: Path) -> Path:
        fetched.append(source.name)
        return root

    def fake_fetch_gaia(root: Path, service: object) -> Path:
        fetched.append(f"gaia from {service}")
        return root

    monkeypatch.setattr(milky_way, "fetch_file", fake_fetch_file)
    monkeypatch.setattr(milky_way, "fetch_gaia", fake_fetch_gaia)
    monkeypatch.setattr(pyvo.dal, "TAPService", lambda url: "the archive")
    invoke("milky-way", "fetch", "--root", str(tmp_path), "--only", "lamost", "--only", "gaia")
    assert fetched == ["lamost", "gaia from the archive"]
    invoke("milky-way", "fetch", "--root", str(tmp_path))
    assert fetched[2:] == ["astronn", "allstar", "lamost", "distmass", "imig", "gaia from the archive"]


def test_paper_figures_draw_from_the_records(result: MilkyWay, mock_run: MockRun, tmp_path: Path) -> None:
    """Every figure of the paper draws from the records, and the command writes them from a data directory."""
    data = tmp_path / "data"
    galaxy, pot = mock_run.prepared.galaxy, mock_run.prepared.potential
    tracer_sum = TracerSumRecord(MOCK_SNAPSHOT, measure_tracer_sum(galaxy, pot, PalicioWindow(20.0, 1.0), n_draws=2))
    write_json(mock_run.analysis, data / "auriga" / result_name(MOCK_SNAPSHOT))
    write_json(tracer_sum, data / "auriga" / tracer_sum_name(MOCK_SNAPSHOT))
    write_json(run_injection(n_stars=300, draws=2), data / "window-injection.json")
    write_json(result, data / "milky-way" / "milky-way.json")
    output = invoke("paper-figures", str(MOCK_SNAPSHOT.halo), "--data-dir", str(data), "--figures-dir", str(tmp_path))
    assert output.count("wrote") == len(FIGURES)
    assert all((tmp_path / f"{name}.pdf").exists() for name in FIGURES)
