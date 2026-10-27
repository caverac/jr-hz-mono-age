/**
 * The record that `experiments milky-way analyze` writes (`assets/data/milky-way/milky-way.json`):
 * the star-count h_Z of the mono-age populations of the Milky Way, with LAMOST and with
 * APOGEE, against the isothermal tracer and the sheet in McMillan (2017), and the check
 * against the pooled heights of Imig et al. (2025).
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
} from '@site/src/components/status/data'

export const MILKY_WAY_PATH = '/data/milky-way/milky-way.json'

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

/** One age bin of Imig et al.: their pooled h_Z, the APOGEE sigma_z of the same ages, the tracer. */
export interface ImigBin {
  readonly ageLo: number
  readonly ageHi: number
  readonly nPopulations: number
  readonly measured: Estimate
  readonly nKin: number
  readonly sigmaZ: Estimate
  readonly tracer: number
  readonly ratio: Estimate
}

export interface MilkyWayData {
  readonly r0: number
  readonly sigmaSurf: number
  readonly k: number
  readonly nBootstrap: number
  readonly surveys: readonly MwSurvey[]
  readonly imig: { readonly bins: readonly ImigBin[]; readonly trend: Trend }
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
  return {
    ageLo: num(bin, 'age_lo'),
    ageHi: num(bin, 'age_hi'),
    nPopulations: num(bin, 'n_populations'),
    measured: estimate(bin, 'measured'),
    nKin: num(bin, 'n_kin'),
    sigmaZ: estimate(bin, 'sigma_z'),
    tracer: num(bin, 'tracer'),
    ratio: estimate(bin, 'ratio')
  }
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
    imig: { bins: list(imig, 'bins').map(parseImigBin), trend: trend(imig, 'trend') }
  }
}
