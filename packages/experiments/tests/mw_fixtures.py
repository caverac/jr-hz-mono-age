"""A small synthetic Milky Way: the six inputs of the analysis, written as FITS files under one root.

The Gaia cylinder holds red giants whose |z| is exponential with a scale that depends on
age, and whose Wesenheit magnitudes follow a red clump at M_W = -1.75 above a flat
background. LAMOST and APOGEE observe a random subset of them, with ages, abundances of the
low-[alpha/Fe] disk, and the phase-space coordinates of stars of the solar annulus. The
distmass ages are 10 percent older than the astroNN ones. The heights and dispersions are
not tied to each other: the files only exercise the reading, matching and measuring code.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table

from experiments.mw_sources import FILE_SOURCES, GAIA_PATH

N_GAIA = 60_000
SEED = 7
BP_RP = 1.2
RC_PEAK = -1.75


def scale_height(age: np.ndarray) -> np.ndarray:
    """Return the vertical scale (kpc) of the stars of *age* Gyr."""
    return np.asarray(0.15 + 0.04 * age)


def _path(root: Path, name: str) -> Path:
    path = root / next(s.path for s in FILE_SOURCES if s.name == name)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_world(root: Path) -> dict[str, np.ndarray]:
    """Write the six inputs under *root* and return the truth of the Gaia stars."""
    rng = np.random.default_rng(SEED)
    age = rng.uniform(0.5, 10.0, N_GAIA)
    z = rng.exponential(scale_height(age)) * rng.choice([-1.0, 1.0], N_GAIA)
    rho = np.sqrt(rng.uniform(0.0, 1.4**2, N_GAIA))
    distance = np.hypot(rho, z)
    b = np.degrees(np.arcsin(z / distance))
    l_deg = rng.uniform(0.0, 360.0, N_GAIA)
    clump = rng.uniform(size=N_GAIA) < 0.6
    m_w = np.where(clump, rng.normal(RC_PEAK, 0.1, N_GAIA), rng.uniform(-2.6, -0.9, N_GAIA))
    parallax = 1.0 / distance
    g = m_w + 1.9 * BP_RP - 5.0 * np.log10(parallax) + 10.0
    source_id = np.arange(1, N_GAIA + 1, dtype=np.int64)
    gaia = Table(
        {
            "source_id": source_id, "ra": l_deg, "dec": b, "l": l_deg, "b": b, "parallax": parallax,
            "parallax_error": parallax / 50.0, "pmra": np.zeros(N_GAIA), "pmdec": np.zeros(N_GAIA),
            "phot_g_mean_mag": g, "bp_rp": np.full(N_GAIA, BP_RP), "ruwe": np.ones(N_GAIA),
        }
    )  # fmt: skip
    gaia_path = root / GAIA_PATH
    gaia_path.parent.mkdir(parents=True, exist_ok=True)
    gaia.write(gaia_path, overwrite=True)

    picked = rng.choice(N_GAIA, 12_000, replace=False)
    fe_h = rng.uniform(-0.4, 0.3, picked.size)
    v_z = rng.normal(0.0, 10.0 + 2.0 * age[picked])
    lamost = Table(
        {
            "RA": rng.uniform(0.0, 360.0, picked.size), "DEC": rng.uniform(-30.0, 30.0, picked.size),
            "GLON": l_deg[picked], "GLAT": b[picked], "RV": v_z, "ERR_RV": np.full(picked.size, 1.0),
            "FEH": fe_h, "AFE": np.full(picked.size, 0.0), "AGE": age[picked], "ERR_AGE": 0.2 * age[picked],
            "G": g[picked], "DISTANCE": rng.uniform(0.1, 0.8, picked.size),
            "ERR_DISTANCE": np.full(picked.size, 0.01), "PMRA": rng.normal(0.0, 5.0, picked.size),
            "ERR_PMRA": np.full(picked.size, 0.05), "PMDEC": rng.normal(0.0, 5.0, picked.size),
            "ERR_PMDEC": np.full(picked.size, 0.05),
        }
    )  # fmt: skip
    lamost.write(_path(root, "lamost"), overwrite=True)

    n_ap = 10_000
    ap = rng.choice(N_GAIA, n_ap, replace=False)
    ap_fe = rng.uniform(-0.4, 0.3, n_ap)
    astronn = Table(
        {
            "APOGEE_ID": np.array([f"2M{k:08d}" for k in range(n_ap)]), "source_id": source_id[ap],
            "TEFF": np.full(n_ap, 4600.0), "LOGG": np.full(n_ap, 2.5), "FE_H": ap_fe, "MG_H": ap_fe,
            "weighted_dist": distance[ap] * 1000.0, "weighted_dist_error": distance[ap] * 10.0,
            "age_lowess_correct": age[ap], "age_total_error": 0.2 * age[ap],
            "galr": rng.uniform(7.6, 8.9, n_ap), "galz": z[ap], "galvr": rng.normal(0.0, 30.0, n_ap),
            "galvt": rng.normal(230.0, 20.0, n_ap), "galvz": rng.normal(0.0, 10.0 + 2.0 * age[ap]),
            "galvr_err": np.full(n_ap, 1.0), "galvz_err": np.full(n_ap, 1.0),
        }
    )  # fmt: skip
    astronn.write(_path(root, "astronn"), overwrite=True)
    ids = astronn["APOGEE_ID"].copy()
    ids[0] = "2MXXXXXXXX"  # a row whose APOGEE_ID differs from astroNN's gets no age
    Table(
        {
            "APOGEE_ID": ids, "AGE_UNCOR_SS": 1.1e9 * age[ap],
            "AGE_ERR": np.tile([0.15, -0.13], (n_ap, 1)),
        }
    ).write(_path(root, "distmass"), overwrite=True)  # fmt: skip
    Table({"EXTRATARG": np.zeros(n_ap, dtype=np.int16)}).write(_path(root, "allstar"), overwrite=True)

    ages = [(1.0, 1.26), (1.26, 1.58), (1.58, 2.0), (2.0, 2.51)]
    rows = [(lo, hi, m_h) for lo, hi in ages for m_h in (-0.25, 0.05)]
    n_imig = len(rows) + 1
    columns = [
        fits.Column(name="ALPHA_M", format="4A", array=np.array(["LOW"] * len(rows) + ["HIGH"])),
        fits.Column(name="M_H", format="D", array=np.array([r[2] for r in rows] + [0.0])),
        fits.Column(name="MIN_AGE", format="D", array=np.array([r[0] for r in rows] + [1.0])),
        fits.Column(name="MAX_AGE", format="D", array=np.array([r[1] for r in rows] + [1.26])),
        fits.Column(name="H_Z", format="D", array=np.array([scale_height(np.array(r[0]))[()] for r in rows] + [0.9])),
        fits.Column(name="H_Z_ERR", format="2D", array=np.full((n_imig, 2), 0.01)),
        fits.Column(name="SMD_0", format="D", array=np.full(n_imig, 1.0)),
        fits.Column(name="SMD_0_ERRS", format="2D", array=np.full((n_imig, 2), 0.05)),
        fits.Column(name="NU_0", format="D", array=np.full(n_imig, 2.0)),
        fits.Column(name="NU_0_ERRS", format="2D", array=np.full((n_imig, 2), 0.1)),
        fits.Column(name="A_FLARE", format="D", array=np.full(n_imig, 0.05)),
    ]
    fits.BinTableHDU.from_columns(columns).writeto(_path(root, "imig"), overwrite=True)
    return {"absz": np.abs(z), "age": age, "m_w": m_w}
