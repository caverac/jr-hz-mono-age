/**
 * The physics that the Concepts pages compute in the browser: a toy disk, the isothermal tracer
 * and the sheet in it, the windowed scale height of `experiments.heights`, and the density of a
 * population for any distribution function of its vertical energy.
 *
 * Units: heights in kpc, velocities in km/s, surface densities in Msun/pc^2, densities in
 * Msun/pc^3, potentials in (km/s)^2.
 */

/** The gravitational constant in pc Msun^-1 (km/s)^2, as `experiments.heights.G_PC`. */
export const G_PC = 4.30091e-3

const PC_PER_KPC = 1000

/**
 * A toy disk: a self-gravitating isothermal layer of surface density `sigmaSurf` and dispersion
 * `sigmaDisk`, in a dark halo of uniform density `rhoHalo` near the plane.
 */
export interface ToyDisk {
  readonly sigmaDisk: number
  readonly sigmaSurf: number
  readonly rhoHalo: number
}

/** A disk like the solar neighborhood: a layer of 55 Msun/pc^2 at 18 km/s in 0.012 Msun/pc^3 of halo. */
export const SOLAR_TOY: ToyDisk = { sigmaDisk: 18, sigmaSurf: 55, rhoHalo: 0.012 }

/** The scale z_0 (kpc) of the sech^2 layer of the disk, sigma^2 / (pi G Sigma). */
export function diskScale(disk: ToyDisk): number {
  return disk.sigmaDisk ** 2 / (Math.PI * G_PC * disk.sigmaSurf) / PC_PER_KPC
}

/** Delta Phi (km/s)^2 of the layer alone at height `z` (kpc): 2 sigma^2 ln cosh(z / z_0). */
export function layerPotential(disk: ToyDisk, z: number): number {
  const x = Math.abs(z) / diskScale(disk)
  // ln cosh x = x + ln(1 + e^-2x) - ln 2, which does not overflow.
  return 2 * disk.sigmaDisk ** 2 * (x + Math.log1p(Math.exp(-2 * x)) - Math.LN2)
}

/** Delta Phi (km/s)^2 of the uniform halo at height `z` (kpc): 2 pi G rho z^2. */
export function haloPotential(disk: ToyDisk, z: number): number {
  return 2 * Math.PI * G_PC * disk.rhoHalo * (z * PC_PER_KPC) ** 2
}

/** Delta Phi (km/s)^2 of the whole toy disk at height `z` (kpc). */
export function toyPotential(disk: ToyDisk, z: number): number {
  return layerPotential(disk, z) + haloPotential(disk, z)
}

/** The midplane density (Msun/pc^3) of the toy disk: Sigma / (2 z_0) of the layer plus the halo. */
export function midplaneDensity(disk: ToyDisk): number {
  return disk.sigmaSurf / (2 * diskScale(disk) * PC_PER_KPC) + disk.rhoHalo
}

/** Delta Phi (km/s)^2 of the harmonic potential with the midplane density of the toy disk. */
export function harmonicPotential(disk: ToyDisk, z: number): number {
  return 2 * Math.PI * G_PC * midplaneDensity(disk) * (z * PC_PER_KPC) ** 2
}

/** The surface density (Msun/pc^2) of all the mass of the toy disk within |z| < `limit` (kpc). */
export function surfaceDensityWithin(disk: ToyDisk, limit: number): number {
  return disk.sigmaSurf * Math.tanh(limit / diskScale(disk)) + 2 * disk.rhoHalo * limit * PC_PER_KPC
}

/** The scale z_0 (kpc) of the sheet sigma^2 / (pi G Sigma) for a dispersion `sigma` (km/s). */
export function sheetScale(sigma: number, sigmaSurf: number): number {
  return sigma ** 2 / (Math.PI * G_PC * sigmaSurf) / PC_PER_KPC
}

/** sech^2(z / z_0), written so that it does not overflow. */
export function sech2(z: number, z0: number): number {
  const e = Math.exp((-2 * Math.abs(z)) / z0)
  return (4 * e) / (1 + e) ** 2
}

/** `n` evenly spaced values from `lo` to `hi`. */
export function linspace(lo: number, hi: number, n: number): readonly number[] {
  return Array.from({ length: n }, (_, i) => lo + ((hi - lo) * i) / (n - 1))
}

/** `n` geometrically spaced values from `lo` to `hi`. */
export function geomspace(lo: number, hi: number, n: number): readonly number[] {
  return linspace(Math.log(lo), Math.log(hi), n).map(Math.exp)
}

/** The heights (kpc) of a profile, as `experiments.heights.Z_GRID`. */
export const Z_GRID = linspace(0, 12, 2401)

/** The trial windows (kpc) of the self-consistent search, as `experiments.heights.WINDOW_GRID`. */
export const WINDOW_GRID = geomspace(0.05, 12, 40)

const BISECTION_STEPS = 30
const SCALE_BRACKET: readonly [number, number] = [1e-4, 1e3]
const MAX_EXPONENT = 700

/** The mean height of an exponential of scale `scale` truncated at `window`. */
export function truncatedMean(scale: number, window: number): number {
  return scale - window / Math.expm1(Math.min(window / scale, MAX_EXPONENT))
}

/**
 * The scale of the exponential on [0, `window`] whose mean height is `meanZ`, or NaN without a
 * root: the maximum-likelihood rate exists only while `meanZ` < `window` / 2.
 */
export function truncatedExpScale(meanZ: number, window: number): number {
  const residual = (scale: number): number => truncatedMean(scale, window) - meanZ
  let [lo, hi] = SCALE_BRACKET
  if (!(residual(lo) < 0 && residual(hi) > 0)) {
    return Number.NaN
  }
  for (let i = 0; i < 80; i += 1) {
    const mid = Math.sqrt(lo * hi)
    if (residual(mid) < 0) {
      lo = mid
    } else {
      hi = mid
    }
  }
  return Math.sqrt(lo * hi)
}

/** The mean height of the profile `rho` on `z` over [0, `limit`], with the profile interpolated at `limit`. */
export function profileMean(rho: readonly number[], z: readonly number[], limit: number): number {
  let mass = 0
  let moment = 0
  for (let i = 1; i < z.length; i += 1) {
    const z0 = z[i - 1]
    if (z0 >= limit) {
      break
    }
    const z1 = Math.min(z[i], limit)
    const r1 =
      z[i] <= limit ? rho[i] : rho[i - 1] + ((rho[i] - rho[i - 1]) * (z1 - z0)) / (z[i] - z0)
    const dz = z1 - z0
    mass += 0.5 * (rho[i - 1] + r1) * dz
    moment += 0.5 * (z0 * rho[i - 1] + z1 * r1) * dz
  }
  return moment / mass
}

/** The windowed h_Z of the profile `rho`: the mean of the fits inside `window` and `window` / 2. */
export function profileHz(rho: readonly number[], z: readonly number[], window: number): number {
  const full = truncatedExpScale(profileMean(rho, z, window), window)
  const half = truncatedExpScale(profileMean(rho, z, window / 2), window / 2)
  return 0.5 * (full + half)
}

/** h_Z and its window L at the smallest root of L = `k` h(L) on the grid of trial windows. */
export function selfConsistentWindow(
  hOfWindow: (window: number) => number,
  k = 2
): { readonly hz: number; readonly window: number } {
  const g = (window: number): number => {
    const h = hOfWindow(window)
    return Number.isFinite(h) ? window - k * h : Number.NEGATIVE_INFINITY
  }
  const first = WINDOW_GRID.findIndex((window) => g(window) >= 0)
  if (first <= 0) {
    return { hz: Number.NaN, window: Number.NaN }
  }
  let lo = WINDOW_GRID[first - 1]
  let hi = WINDOW_GRID[first]
  for (let i = 0; i < BISECTION_STEPS; i += 1) {
    const mid = Math.sqrt(lo * hi)
    if (g(mid) >= 0) {
      hi = mid
    } else {
      lo = mid
    }
  }
  return { hz: hOfWindow(hi), window: hi }
}

/** h_Z and its window of the profile `rho` on `z`, in the self-consistent window. */
export function scaledProfileHz(
  rho: readonly number[],
  z: readonly number[] = Z_GRID
): { readonly hz: number; readonly window: number } {
  return selfConsistentWindow((window) =>
    window > z[z.length - 1] ? Number.NaN : profileHz(rho, z, window)
  )
}

/** The isothermal tracer exp(-Delta Phi / sigma^2) of the potential `phi` (a function of z) on `z`. */
export function tracerProfile(
  phi: (z: number) => number,
  sigma: number,
  z: readonly number[] = Z_GRID
): readonly number[] {
  return z.map((height) => Math.exp(-phi(height) / sigma ** 2))
}

/** A distribution function of the vertical energy E ((km/s)^2) with a scale E0 and a shape q. */
export interface EnergyDf {
  readonly e0: number
  /** exp[-(E / E0)^q]: q = 1 is isothermal, q < 1 has heavier tails, q > 1 a sharper cutoff. */
  readonly q: number
}

/** The value of the distribution function `df` at the energy `energy`. */
function dfValue(df: EnergyDf, energy: number): number {
  return Math.exp(-((energy / df.e0) ** df.q))
}

/** The energy (in units of E0) beyond which the distribution function is negligible. */
function cutoff(df: EnergyDf): number {
  return 40 ** (1 / df.q)
}

const N_VELOCITY = 120

/**
 * The density and the mean v_z^2 of a population with the distribution function `df` in the
 * potential `phi` (values on the heights `z`): rho(z) = 2 int_0^inf f(Phi + v^2/2) dv.
 */
export function dfMoments(
  df: EnergyDf,
  phi: readonly number[]
): { readonly rho: readonly number[]; readonly v2: readonly number[] } {
  const vMax = Math.sqrt(2 * cutoff(df) * df.e0)
  const dv = vMax / (N_VELOCITY - 1)
  const rho: number[] = []
  const v2: number[] = []
  for (const potential of phi) {
    let zeroth = 0
    let second = 0
    for (let j = 0; j < N_VELOCITY; j += 1) {
      const v = j * dv
      const weight = j === 0 || j === N_VELOCITY - 1 ? 0.5 : 1
      const f = dfValue(df, potential + 0.5 * v * v) * weight * dv
      zeroth += f
      second += f * v * v
    }
    rho.push(2 * zeroth)
    v2.push(zeroth > 0 ? second / zeroth : 0)
  }
  return { rho, v2 }
}

/** The trapezoid integral of `y` over `x`. */
export function trapezoid(y: readonly number[], x: readonly number[]): number {
  let total = 0
  for (let i = 1; i < x.length; i += 1) {
    total += 0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1])
  }
  return total
}

/** The mean v_z^2 and the rms height of a population whose density and mean v_z^2 are given on `z`. */
export function populationMoments(
  rho: readonly number[],
  v2: readonly number[],
  z: readonly number[]
): { readonly meanV2: number; readonly rmsZ: number } {
  const mass = trapezoid(rho, z)
  return {
    meanV2:
      trapezoid(
        rho.map((r, i) => r * v2[i]),
        z
      ) / mass,
    rmsZ: Math.sqrt(
      trapezoid(
        rho.map((r, i) => r * z[i] ** 2),
        z
      ) / mass
    )
  }
}

/**
 * The scale E0 of the distribution function of shape `q` whose population-averaged v_z^2 in the
 * potential `phi` on `z` is `sigma`^2, by bisection in ln E0.
 */
export function matchedDf(
  q: number,
  sigma: number,
  phi: readonly number[],
  z: readonly number[]
): EnergyDf {
  let lo = Math.log(sigma ** 2 / 50)
  let hi = Math.log(sigma ** 2 * 50)
  for (let i = 0; i < 24; i += 1) {
    const mid = 0.5 * (lo + hi)
    const { rho, v2 } = dfMoments({ e0: Math.exp(mid), q }, phi)
    if (populationMoments(rho, v2, z).meanV2 < sigma ** 2) {
      lo = mid
    } else {
      hi = mid
    }
  }
  return { e0: Math.exp(0.5 * (lo + hi)), q }
}
