"""Tests of the maps, the star sample and the rendered images of the disks."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from conftest import MockRun
from matplotlib.colors import to_rgb

from experiments.render import (
    AGE_COLORS,
    AGE_GROUPS_GYR,
    AGE_STOPS_GYR,
    AGE_UNIT_GYR,
    GROUP_AGES_GYR,
    HALF_HEIGHT_KPC,
    HALF_WIDTH_KPC,
    POSITION_UNIT_KPC,
    StarMaps,
    age_colormap,
    composite,
    load_maps,
    portrait_figure,
    save_maps,
    star_maps,
    star_sample,
    write_sample,
)
from experiments.snapshot import Galaxy


def stellar_ages(galaxy: Galaxy) -> np.ndarray:
    """Return the age (Gyr) of every star particle of *galaxy*."""
    return galaxy.header.cosmology.lookback_gyr(galaxy.stars.formation_a, galaxy.header.scale_factor)


@pytest.fixture(name="maps", scope="module")
def fixture_maps(mock_run: MockRun) -> StarMaps:
    """Return the maps of the mock galaxy at 60 pixels across."""
    galaxy = mock_run.prepared.galaxy
    return star_maps(galaxy.stars, stellar_ages(galaxy), pixels=60)


def test_maps_conserve_the_mass_of_each_group(mock_run: MockRun, maps: StarMaps) -> None:
    """Each map holds the mass of its age group inside the box, in both projections."""
    galaxy = mock_run.prepared.galaxy
    x, y, z = galaxy.stars.pos.T
    inside = (np.abs(x) < HALF_WIDTH_KPC) & (np.abs(y) < HALF_WIDTH_KPC) & (np.abs(z) < HALF_HEIGHT_KPC)
    pixel_area = (2.0 * HALF_WIDTH_KPC / 60) ** 2
    age = stellar_ages(galaxy)
    assert maps.face_on.shape == (len(AGE_GROUPS_GYR), 60, 60)
    assert maps.edge_on.shape == (len(AGE_GROUPS_GYR), 18, 60)
    for i, (lo, hi) in enumerate(AGE_GROUPS_GYR):
        mass = galaxy.stars.mass[inside & (age >= lo) & (age < hi)].sum()
        assert maps.face_on[i].sum() * pixel_area == pytest.approx(mass)
        assert maps.edge_on[i].sum() * pixel_area == pytest.approx(mass)


def test_maps_round_trip(maps: StarMaps, tmp_path: Path) -> None:
    """Saved maps load back to float32 precision."""
    loaded = load_maps(save_maps(maps, tmp_path / "out" / "maps.npz"))
    np.testing.assert_allclose(loaded.face_on, maps.face_on, rtol=1e-6)
    np.testing.assert_allclose(loaded.edge_on, maps.edge_on, rtol=1e-6)
    assert (loaded.half_width, loaded.half_height) == (maps.half_width, maps.half_height)


def test_star_sample_and_its_files(mock_run: MockRun, tmp_path: Path) -> None:
    """The sample keeps stars within 30 kpc and 10 kpc of the plane, and its binary decodes to them."""
    galaxy = mock_run.prepared.galaxy
    age = stellar_ages(galaxy)
    sample = star_sample(galaxy.stars, age, 500, np.random.default_rng(1))
    assert sample.ages.size == 500
    positions = sample.positions.astype(np.float64) * POSITION_UNIT_KPC
    assert np.all(np.hypot(positions[:, 0], positions[:, 1]) < 30.0 + 1e-3)
    assert np.all(np.abs(positions[:, 2]) < 10.0 + 1e-3)
    everything = star_sample(galaxy.stars, age, 10**9, np.random.default_rng(1))
    assert everything.ages.size < 10**9
    binary, meta = write_sample(sample, "Au0", tmp_path / "au0-stars")
    raw = binary.read_bytes()
    split = 500 * 6  # three int16 positions per star come first
    np.testing.assert_array_equal(np.frombuffer(raw[:split], dtype="<i2").reshape(500, 3), sample.positions)
    np.testing.assert_array_equal(np.frombuffer(raw[split:], dtype="<u2"), sample.ages)
    description = json.loads(meta.read_text())
    assert description["count"] == 500
    assert description["ageUnitGyr"] == AGE_UNIT_GYR
    assert description["ageStopsGyr"] == list(AGE_STOPS_GYR)
    assert description["ageColors"] == list(AGE_COLORS)


def test_age_colormap_runs_from_the_youngest_to_the_oldest_color() -> None:
    """The colormap starts at the first color and ends at the last."""
    cmap = age_colormap()
    np.testing.assert_allclose(cmap(0.0)[:3], to_rgb(AGE_COLORS[0]))
    np.testing.assert_allclose(cmap(1.0)[:3], to_rgb(AGE_COLORS[-1]))


def test_composite(maps: StarMaps) -> None:
    """The image has one rgb triple per pixel in [0, 1]; a young group is blue, and nothing is black."""
    image = composite(maps.face_on, GROUP_AGES_GYR)
    assert image.shape == (60, 60, 3)
    assert 0.0 <= image.min() and image.max() <= 1.0
    young = composite(maps.face_on[:1], GROUP_AGES_GYR[:1])
    center = young[30, 30]
    assert center[2] > center[0]
    assert not composite(np.zeros((1, 4, 4)), (1.0,)).any()


def test_portrait(maps: StarMaps) -> None:
    """The portrait has a face-on and an edge-on panel, with the label, the scale bar and the age key."""
    portrait = portrait_figure(maps, "Au0")
    assert len(portrait.axes) == 2
    texts = [text.get_text() for axes in portrait.axes for text in axes.texts]
    assert "Au0" in texts and "10 kpc" in texts and "age (Gyr)" in texts
    plt.close("all")
