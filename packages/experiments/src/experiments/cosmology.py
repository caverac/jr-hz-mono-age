"""Ages of star particles from their formation scale factors, in flat LambdaCDM.

Auriga records the scale factor at which each star particle formed
(``GFM_StellarFormationTime``). In a flat universe of matter and a cosmological constant,
without radiation, the cosmic time at scale factor a is

    t(a) = 2 / (3 H_0 sqrt(Omega_L)) asinh(sqrt(Omega_L / Omega_m) a^{3/2}),

so the age of a star particle seen in a snapshot at a_now is t(a_now) - t(a).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

GYR_PER_MPC_KMS = 977.792  # 1 Mpc / (1 km/s) in Gyr (Julian year)


@dataclass(frozen=True)
class Cosmology:
    """A flat LambdaCDM cosmology, with H_0 = 100 h km/s/Mpc."""

    hubble: float
    omega_m: float
    omega_lambda: float

    @property
    def _norm(self) -> float:
        """Return 2 / (3 H_0 sqrt(Omega_L)) in Gyr."""
        return float(2.0 * GYR_PER_MPC_KMS / (100.0 * self.hubble) / (3.0 * np.sqrt(self.omega_lambda)))

    @property
    def _root(self) -> float:
        """Return sqrt(Omega_L / Omega_m)."""
        return float(np.sqrt(self.omega_lambda / self.omega_m))

    def cosmic_time(self, scale_factor: FloatArray) -> FloatArray:
        """Return the cosmic time (Gyr) at each scale factor."""
        return np.asarray(self._norm * np.arcsinh(self._root * scale_factor**1.5), dtype=np.float64)

    def lookback_gyr(self, scale_factor: FloatArray, now: float) -> FloatArray:
        """Return the time (Gyr) from each scale factor to the scale factor *now*."""
        return np.asarray(self.cosmic_time(np.array([now]))[0] - self.cosmic_time(scale_factor), dtype=np.float64)

    def scale_factor_at_lookback(self, lookback_gyr: FloatArray, now: float) -> FloatArray:
        """Return the scale factor *lookback_gyr* before the scale factor *now*, the inverse of :meth:`lookback_gyr`."""
        time = self.cosmic_time(np.array([now]))[0] - lookback_gyr
        return np.asarray((np.sinh(time / self._norm) / self._root) ** (2.0 / 3.0), dtype=np.float64)
