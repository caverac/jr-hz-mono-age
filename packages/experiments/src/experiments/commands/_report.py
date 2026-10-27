"""The printed report and the JSON record of an analysis."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

from experiments.analysis import HaloAnalysis
from experiments.populations import AgeBin, AgeRange, ScanRow
from experiments.whole_disk import Ring

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

BIN_HEADER = (
    "    age   med  N_win    sR    sz  J_R,med       h_Z(k h_Z)    window  h_Z(Z_lim)  "
    "sheet  tracer  snapshot        a"
)
RING_HEADER = "         R (kpc)    sz   Sigma  h_Z(Z_lim)  h_Z(k h_Z)  window   sheet  tracer"


def _bin_line(b: AgeBin) -> str:
    """Return the row of one age bin."""
    kin, hts, pred = b.kinematics, b.heights, b.predictions
    return (
        f"  {b.age_lo:2.0f}-{b.age_hi:<2.0f} {b.age_median:5.2f} {hts.n_window:6d} {kin.sigma_r.value:5.1f} "
        f"{kin.sigma_z.value:5.1f} {kin.jr_median.value:6.1f}+/-{kin.jr_median.err:4.1f} "
        f"{hts.measured.value:6.3f}+/-{hts.measured.err:.3f} {hts.window:6.2f} {hts.fixed:11.3f} "
        f"{pred.sheet:6.3f} {pred.tracer:7.3f} {pred.tracer_snapshot:9.3f} "
        f"{b.slopes.measured.value:6.1f}+/-{b.slopes.measured.err:4.1f}"
    )


def _range_line(r: AgeRange) -> str:
    """Return the exponents over one age range."""
    fits = ", ".join(f"{name} {e.value:+.3f}+/-{e.err:.3f}" for name, e in r.exponents.items())
    return f"  {r.lo:g}-{r.hi:g} Gyr: {fits}; eta {r.eta:.2f}, 2(beta_R - beta_z) {r.sheet_exponent:+.3f}"


def _scan_line(label: str, rows: list[ScanRow]) -> str:
    """Return the exponents of a over one scan of the window."""
    return f"  {label} scan: " + ", ".join(
        f"{row.parameter:.2f}: measured {row.measured:+.3f} tracer {row.tracer:+.3f}" for row in rows
    )


def _ring_line(ring: Ring) -> str:
    """Return the row of one ring of the whole disk."""
    h = ring.heights
    return (
        f"  {ring.r_lo:6.1f}-{ring.r_hi:<6.1f} {ring.sigma_z_mid:6.1f} {ring.sigma_surf:7.1f} {h.palicio:11.3f} "
        f"{h.scaled:11.3f} {h.window:7.2f} {h.sheet:7.3f} {h.tracer:7.3f}"
    )


def report_lines(analysis: HaloAnalysis) -> list[str]:
    """Return the printed report of *analysis*."""
    snap, ann, mono, disk = analysis.snapshot, analysis.annulus, analysis.mono_age, analysis.whole_disk
    window = mono.method.window
    return [
        f"Au{snap.halo} level {snap.level} {snap.suite} snapshot {snap.snapshot}",
        f"  annulus {ann.r_lo:g}-{ann.r_hi:g} kpc: R_mid {ann.r_mid:.2f} kpc, v_c {ann.v_circ:.1f} km/s, "
        f"Sigma {ann.sigma_surf:.1f} Msun/pc^2, {ann.n_stars:,} star particles",
        "  Delta Phi (km/s)^2, snapshot / fit: "
        + ", ".join(f"{row.z:.2f} kpc {row.snapshot:.0f}/{row.fitted:.0f}" for row in analysis.potential_check),
        f"  Palicio's window L {window.l_kpc:g} kpc, Z_lim {window.z_lim:g} kpc: h_Z of the annulus "
        f"{mono.hz_all_fixed:.3f} kpc, k_P {mono.k_palicio:.2f}; self-consistent window at k = {mono.method.k:g}",
        BIN_HEADER,
        *(_bin_line(b) for b in mono.bins),
        *(_range_line(r) for r in mono.ranges),
        _scan_line("k", mono.k_scan),
        _scan_line("Z_lim", mono.window_scan),
        f"  whole disk, {0.2 * window.l_kpc:g} <= R < {window.l_kpc:g} kpc:",
        RING_HEADER,
        *(_ring_line(ring) for ring in (*disk.rings, disk.whole)),
    ]


def write_json(record: DataclassInstance, path: Path) -> Path:
    """Write the dataclass *record* as JSON to *path*, creating its directory, and return *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(record), indent=2) + "\n")
    return path
