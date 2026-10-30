/**
 * The record that `experiments milky-way analyze` writes (`assets/data/milky-way/milky-way.json`):
 * the star-count h_Z of the mono-age populations of the Milky Way, with LAMOST and with
 * APOGEE on the astroNN and distmass age scales, against the isothermal tracer and the
 * sheet in McMillan (2017), and the comparison of the heights of Imig et al. (2025) with
 * the star counts and the tracer.
 */

import { median } from '@site/src/components/status/metrics'
import {
  bool,
  type Estimate,
  estimate,
  field,
  list,
  num,
  numbers,
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
  /** The gradient b (1/kpc) of the population fraction exp(b x) toward the Galactic center. */
  readonly gradient: number
  /** h_Z (kpc) of an exponential of the scale of the sheet, through the same counts. */
  readonly sheet: number
  /** h_Z (kpc) of the tracer profile, through the same counts. */
  readonly tracer: number
  readonly ratioTracer: Estimate
  readonly ratioSheet: Estimate
  /** h_Z (kpc) of the vertical Jeans equation with the measured sigma_z(z), through the same counts. */
  readonly jeans: number
  readonly ratioJeans: Estimate
  /** The b of the isothermal tracer of the bin's sigma_z in McMillan (2017). */
  readonly bIsothermal: number
  /** The tracer over h_Z in each potential of the family, in the order of `family`. */
  readonly ratioFamily: readonly number[]
  /** The thin-layer population of Ting & Rix (2026) over h_Z, in McMillan (2017) and in each member. */
  readonly ratioCoeval: number
  readonly ratioFamilyCoeval: readonly number[]
}

/** The rho_0 (Msun/pc^3) and Sigma_1.1 (Msun/pc^2) at which every bin of one survey matches its h_Z. */
export interface Inversion {
  readonly survey: string
  readonly nBins: number
  readonly rho0: Estimate
  readonly sigma11: Estimate
  readonly correlation: number
  readonly chi2Dof: number
  readonly modelRms: number
  readonly binRho0: readonly Estimate[]
}

/** One potential around McMillan (2017): its changes, rho_0 (Msun/pc^3), Sigma_1.1 (Msun/pc^2) and v_c (km/s). */
export interface FamilyMember {
  readonly thinFactor: number
  readonly thinHeight: number
  readonly haloFactor: number
  readonly rho0: number
  readonly sigma11: number
  readonly vCirc: number
  readonly admissible: boolean
  /** d ln h / d ln sigma_z of the exponential fit to the tracer, below 1.2 kpc and over the whole profile. */
  readonly growthWindow: number
  readonly growthWhole: number
}

export interface MwRange {
  readonly lo: number
  readonly hi: number
  readonly nBins: number
  readonly tracer: Trend
  readonly sheet: Trend
  readonly jeans: Trend
  /** eta = d ln h_Z / d ln sigma_z, fitted across the bins with the errors of both. */
  readonly eta: Trend
  /** The age exponents of h_Z and of sigma_z over the bins of the range. */
  readonly betaH: Estimate
  readonly betaZ: Estimate
}

/** What the counts of a survey give exponentials of known scales (kpc), over each scale. */
export interface CountBias {
  readonly scales: readonly number[]
  readonly ratios: readonly number[]
}

export interface MwSurvey {
  readonly name: string
  readonly nGaia: number
  readonly nMatched: number
  readonly bins: readonly MwBin[]
  readonly ranges: readonly MwRange[]
  readonly bias: CountBias
}

/**
 * The four h_Z of an age bin of Imig et al.: their populations pooled by mass and by counts
 * of giants at their Sun, as their model gives them through our cylinder and counts, and
 * our APOGEE star counts of the same ages on the astroNN and the distmass scales.
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

/** One population of Imig et al. against our counts and kinematic stars of its ages and [M/H] (kpc, km/s). */
export interface ImigCell {
  readonly ageLo: number
  readonly ageHi: number
  readonly mhLo: number
  readonly mhHi: number
  /** Their stars in the cell. */
  readonly nStars: number
  /** Our matched stars in the cell. */
  readonly nMatched: number
  readonly theirs: {
    /** Their h_Z at their Sun, as published. */
    readonly sun: Estimate
    /** Their model over the disk of our cylinder, in the continuous window. */
    readonly window: number
    /** Their model through our cylinder and counts; NaN with fewer than `minMatches` matches. */
    readonly counted: Estimate
    /** The exponential fit to their model through our cylinder and counts, as ours is fit. */
    readonly exponential: Estimate
  }
  /** Our counts; NaN, and an empty shape, with fewer than `minMatches` matches. */
  readonly ours: {
    readonly windowed: Estimate
    /** The exponential fit to our counts below the edge of the counts, without a window. */
    readonly exponential: Estimate
    readonly gradient: number
    /** Our counts over their model in each slice of |z|, each normalized to its sum. */
    readonly shape: readonly number[]
  }
  /** The kinematic stars of the cell near the plane, and the tracer of their sigma_z through our counts. */
  readonly kinematics: {
    readonly nKin: number
    readonly sigmaZ: Estimate
    readonly tracer: Estimate
  }
  readonly ratios: {
    /** Our windowed h_Z over their model through our counts. */
    readonly counted: Estimate
    /** Our exponential fit over theirs, both through our counts. */
    readonly exponential: Estimate
    /** The tracer over our windowed h_Z. */
    readonly tracer: Estimate
  }
}

/**
 * The slope of ln h_Z (ln sigma_z) per dex of [M/H] at fixed age, one intercept per age bin,
 * in the same cells.
 */
export interface MetallicitySlopes {
  /** Our windowed h_Z. */
  readonly counts: Trend
  readonly countsExponential: Trend
  /** Their model through our counts. */
  readonly imig: Trend
  readonly imigSun: Trend
  readonly tracer: Trend
  readonly sigmaZ: Trend
}

/**
 * The windowed h_Z of the tracer of `sigmaZ`, the exponential fit to it below the edge of the
 * counts, and the exponential fit to the whole profile.
 */
export interface TracerEstimator {
  readonly sigmaZ: number
  readonly windowed: number
  readonly exponential: number
  readonly exponentialWhole: number
}

/**
 * The slope of the logarithm of a ratio of the cells per ln age at fixed [M/H], and per dex
 * of [M/H] at fixed age, each with one intercept per bin of the other.
 */
export interface RatioTrends {
  readonly age: Trend
  readonly metallicity: Trend
}

/** The trends of the three ratios of a cell. */
export interface CellTrends {
  readonly counted: RatioTrends
  readonly exponential: RatioTrends
  readonly tracer: RatioTrends
}

/** The ratios of a cell, by the names of `CellTrends`. */
export type CellRatio = keyof CellTrends

/** Every cell and, over the measured ones, the median ratio, the trends of the ratios and the slopes of the heights. */
export interface ImigCells {
  /** The matched stars that split the counts into populations. */
  readonly nMatches: number
  readonly minMatches: number
  readonly cells: readonly ImigCell[]
  readonly nMeasured: number
  readonly medianRatio: number
  readonly trends: CellTrends
  readonly slopes: MetallicitySlopes
  readonly dispersion: DispersionSlopes
}

/**
 * The slope of ln h_Z per ln sigma_z of the cell's kinematic stars at fixed [M/H], one
 * intercept per [M/H] bin, in the same cells.
 */
export interface DispersionSlopes {
  /** The exponential fit to our counts. */
  readonly countsExponential: Trend
  /** The exponential fit to their model through our counts. */
  readonly imigExponential: Trend
  /** Their published h_Z at their Sun. */
  readonly imigSun: Trend
}

/** The value of `jkHi` with which the package marks a targeting box without a red limit. */
export const JK_NO_RED_LIMIT = 99

/** A range of H and (J - K_s)_0 inside the targeting limits of every star kept. */
export interface TargetingBox {
  readonly hLo: number
  readonly hHi: number
  readonly jkLo: number
  readonly jkHi: number
}

/** The cells measured with only the matches of one targeting box. */
export interface TargetedCells {
  readonly box: TargetingBox
  readonly cells: ImigCells
}

export interface MilkyWayData {
  readonly r0: number
  readonly sigmaSurf: number
  readonly k: number
  readonly nBootstrap: number
  /** The |z| slices of the counts (kpc). */
  readonly sliceEdges: readonly number[]
  readonly surveys: readonly MwSurvey[]
  readonly imig: {
    readonly bins: readonly ImigBin[]
    readonly trends: Readonly<Record<ImigHeight, Trend>>
  }
  readonly imigCells: ImigCells
  readonly imigCellsTargeting: readonly TargetedCells[]
  /** The cells with the LAMOST counts, whose ages are not on the scale of Imig et al. */
  readonly imigCellsLamost: ImigCells
  readonly estimators: readonly TracerEstimator[]
  /** The potentials around McMillan (2017) whose tracers every bin also carries. */
  readonly family: readonly FamilyMember[]
  /** The rho_0 and Sigma_1.1 of each survey with the tracer, and with the thin-layer population. */
  readonly inversions: readonly Inversion[]
  readonly inversionsCoeval: readonly Inversion[]
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
    gradient: num(heights, 'gradient'),
    sheet: num(predictions, 'sheet'),
    tracer: num(predictions, 'tracer'),
    ratioTracer: estimate(ratios, 'tracer'),
    ratioSheet: estimate(ratios, 'sheet'),
    jeans: num(predictions, 'jeans'),
    ratioJeans: estimate(ratios, 'jeans'),
    bIsothermal: num(predictions, 'b_isothermal'),
    ratioFamily: numbers(ratios, 'family'),
    ratioCoeval: num(ratios, 'coeval'),
    ratioFamilyCoeval: numbers(ratios, 'family_coeval')
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
    jeans: trend(trends, 'jeans'),
    eta: trend(range, 'eta'),
    betaH: estimate(field(range, 'exponents'), 'hz'),
    betaZ: estimate(field(range, 'exponents'), 'sigma_z')
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
    theirs: {
      sun: estimate(field(cell, 'theirs'), 'sun'),
      window: num(field(cell, 'theirs'), 'window'),
      counted: estimate(field(cell, 'theirs'), 'counted'),
      exponential: estimate(field(cell, 'theirs'), 'exponential')
    },
    ours: {
      windowed: estimate(field(cell, 'ours'), 'windowed'),
      exponential: estimate(field(cell, 'ours'), 'exponential'),
      gradient: num(field(cell, 'ours'), 'gradient'),
      shape: numbers(field(cell, 'ours'), 'shape')
    },
    kinematics: {
      nKin: num(field(cell, 'kinematics'), 'n_kin'),
      sigmaZ: estimate(field(cell, 'kinematics'), 'sigma_z'),
      tracer: estimate(field(cell, 'kinematics'), 'tracer')
    },
    ratios: {
      counted: estimate(field(cell, 'ratios'), 'counted'),
      exponential: estimate(field(cell, 'ratios'), 'exponential'),
      tracer: estimate(field(cell, 'ratios'), 'tracer')
    }
  }
}

function parseRatioTrends(trends: unknown, key: string): RatioTrends {
  return {
    age: trend(field(trends, key), 'age'),
    metallicity: trend(field(trends, key), 'metallicity')
  }
}

function parseImigCells(cells: unknown): ImigCells {
  const trends = field(cells, 'trends')
  return {
    nMatches: num(cells, 'n_matches'),
    minMatches: num(cells, 'min_matches'),
    cells: list(cells, 'cells').map(parseImigCell),
    nMeasured: num(cells, 'n_measured'),
    medianRatio: num(cells, 'median_ratio'),
    trends: {
      counted: parseRatioTrends(trends, 'counted'),
      exponential: parseRatioTrends(trends, 'exponential'),
      tracer: parseRatioTrends(trends, 'tracer')
    },
    slopes: {
      counts: trend(field(cells, 'slopes'), 'counts'),
      countsExponential: trend(field(cells, 'slopes'), 'counts_exponential'),
      imig: trend(field(cells, 'slopes'), 'imig'),
      imigSun: trend(field(cells, 'slopes'), 'imig_sun'),
      tracer: trend(field(cells, 'slopes'), 'tracer'),
      sigmaZ: trend(field(cells, 'slopes'), 'sigma_z')
    },
    dispersion: {
      countsExponential: trend(field(cells, 'dispersion'), 'counts_exponential'),
      imigExponential: trend(field(cells, 'dispersion'), 'imig_exponential'),
      imigSun: trend(field(cells, 'dispersion'), 'imig_sun')
    }
  }
}

/** The selection of `box`, as the pages print it. */
export function boxText(box: TargetingBox): string {
  const red = box.jkHi >= JK_NO_RED_LIMIT ? '' : ` < ${box.jkHi}`
  return `$${box.hLo} < H < ${box.hHi}$, $${box.jkLo} < (J-K_s)_0${red}$`
}

/** The measured cells whose [M/H] center lies in [lo, hi). */
export function measuredIn(cells: ImigCells, lo: number, hi: number): readonly ImigCell[] {
  return cells.cells
    .filter((c) => isMeasured(cells, c))
    .filter((c) => 0.5 * (c.mhLo + c.mhHi) >= lo && 0.5 * (c.mhLo + c.mhHi) < hi)
}

/**
 * The median of the ratio `ratio` over the measured cells whose [M/H] center lies in [lo, hi)
 * and where it is finite, and their number.
 */
export function cellMedian(
  cells: ImigCells,
  lo: number,
  hi: number,
  ratio: CellRatio = 'counted'
): { median: number; n: number } {
  const ratios = measuredIn(cells, lo, hi)
    .map((c) => c.ratios[ratio].value)
    .filter((r) => Number.isFinite(r))
  return { median: median(ratios), n: ratios.length }
}

/** The ranges of [M/H] that the pages summarize the cells over: metal-poor, near solar, metal-rich. */
export const METALLICITY_RANGES: readonly (readonly [number, number])[] = [
  [Number.NEGATIVE_INFINITY, -0.1],
  [-0.1, 0.1],
  [0.1, Number.POSITIVE_INFINITY]
]

/** The ranges of `METALLICITY_RANGES` as the tables head them. */
export const METALLICITY_LABELS: readonly string[] = [
  '$[\\mathrm{M/H}] < -0.1$',
  '$-0.1$ to $0.1$',
  '$> 0.1$'
]

/**
 * The power of sigma_z with which the exponential fit to the tracer grows between the first
 * and the last sigma_z (km/s) of the estimators, NaN without two of them.
 */
export function tracerGrowth(estimators: readonly TracerEstimator[]): {
  readonly slope: number
  readonly lo: number
  readonly hi: number
} {
  const first = estimators[0]
  const last = estimators[estimators.length - 1]
  if (first === undefined || last === undefined || first === last) {
    return { slope: Number.NaN, lo: Number.NaN, hi: Number.NaN }
  }
  return {
    slope: Math.log(last.exponential / first.exponential) / Math.log(last.sigmaZ / first.sigmaZ),
    lo: first.sigmaZ,
    hi: last.sigmaZ
  }
}

/** Whether `cell` has enough matches for our h_Z, and a finite ratio. */
export function isMeasured(cells: ImigCells, cell: ImigCell): boolean {
  return cell.nMatched >= cells.minMatches && Number.isFinite(cell.ratios.counted.value)
}

/** Check and read the record of the Milky Way test. */
/** Check and read one survey of a Milky Way record. */
export function parseSurvey(survey: unknown): MwSurvey {
  return {
    name: str(survey, 'name'),
    nGaia: num(survey, 'n_gaia'),
    nMatched: num(survey, 'n_matched'),
    bins: list(survey, 'bins').map(parseBin),
    ranges: list(survey, 'ranges').map(parseRange),
    bias: {
      scales: numbers(field(survey, 'bias'), 'scales'),
      ratios: numbers(field(survey, 'bias'), 'ratios')
    }
  }
}

function parseInversion(inversion: unknown): Inversion {
  return {
    survey: str(inversion, 'survey'),
    nBins: num(inversion, 'n_bins'),
    rho0: estimate(inversion, 'rho0'),
    sigma11: estimate(inversion, 'sigma_11'),
    correlation: num(inversion, 'correlation'),
    chi2Dof: num(inversion, 'chi2_dof'),
    modelRms: num(inversion, 'model_rms'),
    binRho0: list(inversion, 'bin_rho0').map((item) => ({
      value: num(item, 'value'),
      err: num(item, 'err')
    }))
  }
}

export function parseMilkyWay(record: unknown): MilkyWayData {
  const imig = field(record, 'imig')
  return {
    r0: num(record, 'r0_kpc'),
    sigmaSurf: num(record, 'sigma_surf'),
    k: num(record, 'k'),
    nBootstrap: num(record, 'n_bootstrap'),
    sliceEdges: numbers(record, 'slice_edges_kpc'),
    surveys: list(record, 'surveys').map(parseSurvey),
    imig: {
      bins: list(imig, 'bins').map(parseImigBin),
      trends: byHeight(field(imig, 'trends'), trend)
    },
    imigCells: parseImigCells(field(record, 'imig_cells')),
    imigCellsTargeting: list(record, 'imig_cells_targeting').map((targeted) => {
      const box = field(targeted, 'box')
      return {
        box: {
          hLo: num(box, 'h_lo'),
          hHi: num(box, 'h_hi'),
          jkLo: num(box, 'jk_lo'),
          jkHi: num(box, 'jk_hi')
        },
        cells: parseImigCells(field(targeted, 'cells'))
      }
    }),
    imigCellsLamost: parseImigCells(field(record, 'imig_cells_lamost')),
    estimators: list(record, 'estimators').map((e) => ({
      sigmaZ: num(e, 'sigma_z'),
      windowed: num(e, 'windowed'),
      exponential: num(e, 'exponential'),
      exponentialWhole: num(e, 'exponential_whole')
    })),
    family: list(record, 'family').map((member) => {
      const params = field(member, 'params')
      const growth = field(member, 'growth')
      return {
        thinFactor: num(params, 'thin_factor'),
        thinHeight: num(params, 'thin_height_kpc'),
        haloFactor: num(params, 'halo_factor'),
        rho0: num(member, 'rho0'),
        sigma11: num(member, 'sigma_11'),
        vCirc: num(member, 'v_circ'),
        admissible: bool(member, 'admissible'),
        growthWindow: num(growth, 'window'),
        growthWhole: num(growth, 'whole')
      }
    }),
    inversions: list(record, 'inversions').map(parseInversion),
    inversionsCoeval: list(record, 'inversions_coeval').map(parseInversion)
  }
}
