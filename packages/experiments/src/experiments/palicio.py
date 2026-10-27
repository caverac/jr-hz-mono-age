"""The windows in which Palicio et al. (2025) measured h_Z of the Auriga disks.

Palicio et al. (2025, A&A 695, A193, Table 1) fit h_Z of each galaxy within |Z| < Z_lim
and |Z| < Z_lim / 2, over 0.2 L_kpc <= R < L_kpc, where L_kpc also sets the X-Y range of
their maps.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PalicioWindow:
    """The radial range L_kpc and the vertical window Z_lim (kpc) of one galaxy."""

    l_kpc: float
    z_lim: float


PALICIO_WINDOWS: dict[int, PalicioWindow] = {
    2: PalicioWindow(30.0, 2.5),
    3: PalicioWindow(20.0, 1.0),
    16: PalicioWindow(20.0, 1.0),
    19: PalicioWindow(15.0, 0.5),
    23: PalicioWindow(20.0, 1.0),
    24: PalicioWindow(20.0, 1.0),
    27: PalicioWindow(20.0, 1.0),
}
