"""Tests of the family of Milky Way potentials around McMillan (2017)."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.mw_dynamics import R0_KPC, mcmillan17, surface_density
from experiments.mw_potentials import (
    GROWTH_SIGMAS_KMS,
    RHO0,
    SIGMA_11,
    FamilyParams,
    family_member,
    member_potential,
    potential_family,
)


def test_the_unchanged_member_is_mcmillan17() -> None:
    """With no change, the member has the density and the surface density of McMillan (2017) as AGAMA ships it."""
    pot, reference = member_potential(FamilyParams(1.0, 0.3, 1.0)), mcmillan17()
    point = np.array([[R0_KPC, 0.0, 0.0]])
    assert float(pot.density(point)[0]) == pytest.approx(float(reference.density(point)[0]), rel=1e-6)
    member, _ = family_member(FamilyParams(1.0, 0.3, 1.0))
    assert member.rho0 == pytest.approx(0.1173, abs=5e-4)
    assert member.sigma_11 == pytest.approx(surface_density(reference, R0_KPC), rel=1e-6)
    assert not member.admissible


def test_admissible_members_match_both_measurements() -> None:
    """A member is admissible exactly when its rho_0 and its Sigma_1.1 lie within one error of the measurements."""
    family = potential_family()
    assert len(family) == 36
    for member, model in family:
        inside = abs(member.rho0 - RHO0[0]) <= RHO0[1] and abs(member.sigma_11 - SIGMA_11[0]) <= SIGMA_11[1]
        assert member.admissible == inside
        assert model.sigma_surf == pytest.approx(member.sigma_11)
    assert sum(member.admissible for member, _ in family) > 0


def test_lighter_disks_lower_rho0_and_the_tracer_grows_faster_than_harmonic() -> None:
    """Less thin disk lowers rho_0; over 10-25 km/s the exponential fits grow faster than sigma_z."""
    light, _ = family_member(FamilyParams(0.7, 0.3, 1.0))
    heavy, _ = family_member(FamilyParams(1.0, 0.3, 1.0))
    assert light.rho0 < heavy.rho0
    assert GROWTH_SIGMAS_KMS == (10.0, 25.0)
    for member in (light, heavy):
        assert member.growth.window > 1.0 and member.growth.whole > 1.0
