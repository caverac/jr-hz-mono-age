/**
 * The record that `experiments milky-way analyze` writes (`assets/data/milky-way/milky-way.json`):
 * the star-count h_Z of the mono-age populations of the Milky Way, with LAMOST and with
 * APOGEE on the astroNN and distmass age scales, against the isothermal tracer and the
 * sheet in McMillan (2017), and the comparison of the heights of Imig et al. (2025) with
 * the star counts and the tracer.
 */

import {
  type Estimate,
  estimate,
  field,
  list,
  num,
  str,
  type Trend,
  trend
} from '@site/src/components/status/record'

export const MILKY_WAY_PATH = '/data/milky-way/milky-way.json'

/** The name of every survey of the record, as the pages print it. */
export const SURVEY_NAMES: Readonly<Record<string, string>> = {
  lamost: 'LAMOST',
  apogee: 'APOGEE (astroNN)',
  apogee_distmass: 'APOGEE (distmass)'
}

/** The name that the pages print for the survey `name` of the record. */
export function surveyName(name: string): string {
  return SURVEY_NAMES[name] ?? name
}

/** One mono-age population of the Milky Way. */
export interface MwBin {
  readonly ageLo: number
  readonly ageHi: number
  readonly ageMedian: number
  readonly nKin: number
  readonly sigmaZ: Estimate
  readonly jrMedian: Estimate
  /** h_Z (kpc) of the star counts in the self-consistent window. */
  readonly measured: Estimate
  readonly window: number
  readonly nMatched: number
  /** The fraction of bootstrap draws without an h_Z. */
  readonly failed: number
  readonly sheet: number
  readonly tracer: number
  readonly ratioTracer: Estimate
  readonly ratioSheet: Estimate
}

export interface MwRange {
  readonly lo: number
  readonly hi: number
  readonly nBins: number
  readonly tracer: Trend
  readonly sheet: Trend
  readonly eta: number
}

export interface MwSurvey {
  readonly name: string
  readonly nGaia: number
  readonly nMatched: number
  readonly bins: readonly MwBin[]
  readonly ranges: readonly MwRange[]
}

/**
 * The four h_Z of an age bin of Imig et al.: their populations pooled by mass and by counts
 * of giants at R_0, and our APOGEE star counts of the same ages on the astroNN and the
 * distmass scales.
 */
export type ImigHeight = 'imigMass' | 'imigCounts' | 'countsAstronn' | 'countsDistmass'

/** The heights of an Imig et al. bin in the order the pages list them, with their names. */
export const IMIG_HEIGHTS: readonly { readonly height: ImigHeight; readonly name: string }[] = [
  { height: 'imigMass', name: 'Imig et al., by mass' },
  { height: 'imigCounts', name: 'Imig et al., by counts' },
  { height: 'countsAstronn', name: 'APOGEE (astroNN) counts' },
  { height: 'countsDistmass', name: 'APOGEE (distmass) counts' }
]

/** One age bin of Imig et al.: its four h_Z (kpc), the sigma_z and tracer of its ages, the tracer over each h_Z. */
export interface ImigBin {
  readonly ageLo: number
  readonly ageHi: number
  readonly nPopulations: number
  readonly heights: Readonly<Record<ImigHeight, Estimate>>
  /** The APOGEE kinematic stars of the same ages on the distmass scale. */
  readonly nKin: number
  readonly sigmaZ: Estimate
  readonly tracer: number
  readonly ratios: Readonly<Record<ImigHeight, Estimate>>
}

/** One population of Imig et al. against our counts of its ages and [M/H]: their h_Z at R_0, ours (kpc), the ratio. */
export interface ImigCell {
  readonly ageLo: number
  readonly ageHi: number
  readonly mhLo: number
  readonly mhHi: number
  /** Their stars in the cell. */
  readonly nStars: number
  /** Our matched stars in the cell. */
  readonly nMatched: number
  readonly imig: Estimate
  /** NaN with fewer than `minMatches` matches. */
  readonly counts: Estimate
  /** Our counts over their h_Z. */
  readonly ratio: Estimate
}

/** Every cell and, over the measured ones, the median ratio and the slope of its logarithm per dex of [M/H]. */
export interface ImigCells {
  readonly minMatches: number
  readonly cells: readonly ImigCell[]
  readonly nMeasured: number
  readonly medianRatio: number
  readonly trend: Trend
}

export interface MilkyWayData {
  readonly r0: number
  readonly sigmaSurf: number
  readonly k: number
  readonly nBootstrap: number
  readonly surveys: readonly MwSurvey[]
  readonly imig: {
    readonly bins: readonly ImigBin[]
    readonly trends: Readonly<Record<ImigHeight, Trend>>
  }
  readonly imigCells: ImigCells
}

/** The age of an Imig et al. bin (Gyr): the geometric mean of its edges, as in the trends of the record. */
export function imigAge(bin: ImigBin): number {
  return Math.sqrt(bin.ageLo * bin.ageHi)
}

/** The value of every height in `value`, under the keys that the record uses. */
function byHeight<T>(
  value: unknown,
  read: (item: unknown, key: string) => T
): Readonly<Record<ImigHeight, T>> {
  return {
    imigMass: read(value, 'imig_mass'),
    imigCounts: read(value, 'imig_counts'),
    countsAstronn: read(value, 'counts_astronn'),
    countsDistmass: read(value, 'counts_distmass')
  }
}

function parseBin(bin: unknown): MwBin {
  const kinematics = field(bin, 'kinematics')
  const heights = field(bin, 'heights')
  const predictions = field(bin, 'predictions')
  const ratios = field(bin, 'ratios')
  return {
    ageLo: num(bin, 'age_lo'),
    ageHi: num(bin, 'age_hi'),
    ageMedian: num(bin, 'age_median'),
    nKin: num(kinematics, 'n_kin'),
    sigmaZ: estimate(kinematics, 'sigma_z'),
    jrMedian: estimate(kinematics, 'jr_median'),
    measured: estimate(heights, 'measured'),
    window: num(heights, 'window'),
    nMatched: num(heights, 'n_matched'),
    failed: num(heights, 'failed'),
    sheet: num(predictions, 'sheet'),
    tracer: num(predictions, 'tracer'),
    ratioTracer: estimate(ratios, 'tracer'),
    ratioSheet: estimate(ratios, 'sheet')
  }
}

function parseRange(range: unknown): MwRange {
  const trends = field(range, 'trends')
  return {
    lo: num(range, 'lo'),
    hi: num(range, 'hi'),
    nBins: num(range, 'n_bins'),
    tracer: trend(trends, 'tracer'),
    sheet: trend(trends, 'sheet'),
    eta: num(range, 'eta')
  }
}

function parseImigBin(bin: unknown): ImigBin {
  const kinematics = field(bin, 'kinematics')
  return {
    ageLo: num(bin, 'age_lo'),
    ageHi: num(bin, 'age_hi'),
    nPopulations: num(bin, 'n_populations'),
    heights: byHeight(field(bin, 'heights'), estimate),
    nKin: num(kinematics, 'n_kin'),
    sigmaZ: estimate(kinematics, 'sigma_z'),
    tracer: num(kinematics, 'tracer'),
    ratios: byHeight(field(bin, 'ratios'), estimate)
  }
}

function parseImigCell(cell: unknown): ImigCell {
  return {
    ageLo: num(cell, 'age_lo'),
    ageHi: num(cell, 'age_hi'),
    mhLo: num(cell, 'm_h_lo'),
    mhHi: num(cell, 'm_h_hi'),
    nStars: num(cell, 'n_stars'),
    nMatched: num(cell, 'n_matched'),
    imig: estimate(cell, 'imig'),
    counts: estimate(cell, 'counts'),
    ratio: estimate(cell, 'ratio')
  }
}

function parseImigCells(cells: unknown): ImigCells {
  return {
    minMatches: num(cells, 'min_matches'),
    cells: list(cells, 'cells').map(parseImigCell),
    nMeasured: num(cells, 'n_measured'),
    medianRatio: num(cells, 'median_ratio'),
    trend: trend(cells, 'trend')
  }
}

/** Whether `cell` has enough matches for our h_Z, and a finite ratio. */
export function isMeasured(cells: ImigCells, cell: ImigCell): boolean {
  return cell.nMatched >= cells.minMatches && Number.isFinite(cell.ratio.value)
}

/** Check and read the record of the Milky Way test. */
export function parseMilkyWay(record: unknown): MilkyWayData {
  const imig = field(record, 'imig')
  return {
    r0: num(record, 'r0_kpc'),
    sigmaSurf: num(record, 'sigma_surf'),
    k: num(record, 'k'),
    nBootstrap: num(record, 'n_bootstrap'),
    surveys: list(record, 'surveys').map((survey) => ({
      name: str(survey, 'name'),
      nGaia: num(survey, 'n_gaia'),
      nMatched: num(survey, 'n_matched'),
      bins: list(survey, 'bins').map(parseBin),
      ranges: list(survey, 'ranges').map(parseRange)
    })),
    imig: {
      bins: list(imig, 'bins').map(parseImigBin),
      trends: byHeight(field(imig, 'trends'), trend)
    },
    imigCells: parseImigCells(field(record, 'imig_cells'))
  }
}
