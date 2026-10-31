/**
 * The records the Concepts pages read for one halo: the expansions of its potential and the fit
 * of every population (`experiments auriga concepts`), and the check of the fitted potential
 * against the snapshot's (`experiments auriga analyze`).
 */

import type { Halo } from '@site/src/components/render/halos'
import type { Loaded } from '@site/src/components/render/useSampleMeta'
import { field, list, num, numbers } from '@site/src/components/status/record'
import useRecord from '@site/src/components/status/useRecord'

/** The nodes (kpc) of the CylSpline in R and z, the radii of the Multipole, and its l_max. */
export interface ExpansionGrids {
  readonly cylsplineR: readonly number[]
  readonly cylsplineZ: readonly number[]
  readonly multipoleR: readonly number[]
  readonly multipoleLmax: number
}

/** A quantity of the whole potential and of its two expansions. */
export interface Split {
  readonly total: readonly number[]
  readonly cylspline: readonly number[]
  readonly multipole: readonly number[]
}

/** What the fit of the scale height of one age bin sees; profiles of unit mass below 12 kpc. */
export interface PopulationFit {
  readonly ageLo: number
  readonly ageHi: number
  readonly nStars: number
  readonly sigmaZ: number
  readonly hz: number
  readonly window: number
  readonly edges: readonly number[]
  readonly density: readonly number[]
  readonly z: readonly number[]
  readonly tracer: readonly number[]
  readonly jeansSigma: readonly number[]
  readonly jeans: readonly number[]
  readonly sheet: readonly number[]
  readonly windows: readonly number[]
  readonly heights: readonly number[]
}

export interface ConceptsRecord {
  readonly rMid: number
  readonly sigmaSurf: number
  readonly grids: ExpansionGrids
  /** Heights (kpc) of the vertical structure at R_mid. */
  readonly z: readonly number[]
  /** Delta Phi ((km/s)^2) at R_mid. */
  readonly phi: Split
  /** The vertical force K_z = dPhi/dz ((km/s)^2/kpc) at R_mid. */
  readonly force: Split
  /** Radii (kpc) of the rotation curve. */
  readonly r: readonly number[]
  /** Circular velocities (km/s). */
  readonly vCirc: Split
  readonly mapR: readonly number[]
  readonly mapZ: readonly number[]
  /** log10 of the density (Msun/pc^3) on mapR x mapZ, rows in R. */
  readonly logCylspline: readonly (readonly number[])[]
  readonly logMultipole: readonly (readonly number[])[]
  readonly bins: readonly PopulationFit[]
}

/** A list of rows of numbers, with NaN where the record holds null. */
function grid(value: unknown, key: string): readonly (readonly number[])[] {
  return list(value, key).map((row) => numbers({ row }, 'row'))
}

function parseBin(bin: unknown): PopulationFit {
  return {
    ageLo: num(bin, 'age_lo'),
    ageHi: num(bin, 'age_hi'),
    nStars: num(bin, 'n_stars'),
    sigmaZ: num(bin, 'sigma_z'),
    hz: num(bin, 'hz'),
    window: num(bin, 'window'),
    edges: numbers(bin, 'edges'),
    density: numbers(bin, 'density'),
    z: numbers(bin, 'z'),
    tracer: numbers(bin, 'tracer'),
    jeansSigma: numbers(bin, 'jeans_sigma'),
    jeans: numbers(bin, 'jeans'),
    sheet: numbers(bin, 'sheet'),
    windows: numbers(bin, 'windows'),
    heights: numbers(bin, 'heights')
  }
}

export function parseConcepts(record: unknown): ConceptsRecord {
  const grids = field(record, 'grids')
  const vertical = field(record, 'vertical')
  const rotation = field(record, 'rotation')
  const map = field(record, 'meridional')
  return {
    rMid: num(record, 'r_mid'),
    sigmaSurf: num(record, 'sigma_surf'),
    grids: {
      cylsplineR: numbers(grids, 'cylspline_r'),
      cylsplineZ: numbers(grids, 'cylspline_z'),
      multipoleR: numbers(grids, 'multipole_r'),
      multipoleLmax: num(grids, 'multipole_lmax')
    },
    z: numbers(vertical, 'z'),
    phi: {
      total: numbers(vertical, 'phi_total'),
      cylspline: numbers(vertical, 'phi_cylspline'),
      multipole: numbers(vertical, 'phi_multipole')
    },
    force: {
      total: numbers(vertical, 'force_total'),
      cylspline: numbers(vertical, 'force_cylspline'),
      multipole: numbers(vertical, 'force_multipole')
    },
    r: numbers(rotation, 'r'),
    vCirc: {
      total: numbers(rotation, 'total'),
      cylspline: numbers(rotation, 'cylspline'),
      multipole: numbers(rotation, 'multipole')
    },
    mapR: numbers(map, 'r'),
    mapZ: numbers(map, 'z'),
    logCylspline: grid(map, 'log_cylspline'),
    logMultipole: grid(map, 'log_multipole'),
    bins: list(record, 'bins').map(parseBin)
  }
}

/** The median Delta Phi ((km/s)^2) of the snapshot and of the fit in one |z| slice of the annulus. */
export interface SliceCheck {
  readonly z: number
  readonly snapshot: number
  readonly fitted: number
}

export function parseCheck(record: unknown): readonly SliceCheck[] {
  return list(record, 'potential_check').map((slice) => ({
    z: num(slice, 'z'),
    snapshot: num(slice, 'snapshot'),
    fitted: num(slice, 'fitted')
  }))
}

/** The Concepts record of `halo`. */
export function useConcepts(halo: Halo): Loaded<ConceptsRecord> {
  return useRecord(`/data/auriga/${halo.stem}-concepts.json`, parseConcepts)
}

/** The check of the fitted potential of `halo` against its snapshot. */
export function usePotentialCheck(halo: Halo): Loaded<readonly SliceCheck[]> {
  return useRecord(`/data/auriga/${halo.stem}.json`, parseCheck)
}
