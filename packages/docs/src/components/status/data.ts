/**
 * The records that the status pages read, as written by the Python package:
 * `experiments auriga analyze`, `experiments auriga tracer-sum` and
 * `experiments auriga equilibrium` (one record each per halo, under `assets/data/auriga/`),
 * `experiments auriga mock-check` (`assets/data/auriga/mock-check.json`) and
 * `experiments window-injection` (`assets/data/window-injection.json`). Every number on
 * the status pages is computed from these records, so that the pages cannot claim more
 * than the data written by the package. A quantity that could not be measured is `null`
 * in the JSON and NaN here.
 */

import { HALOS } from '@site/src/components/render/halos'
import { type MotionBin, parseEquilibrium } from '@site/src/components/status/equilibrium'
import {
  type Estimate,
  estimate,
  fetchJson,
  field,
  list,
  num,
  numbers,
  RecordError,
  str,
  type Trend,
  trend
} from '@site/src/components/status/record'
import { parseTracerSum, type TracerSum } from '@site/src/components/status/tracerSum'

export type { Estimate, Trend } from '@site/src/components/status/record'

/** The predictions of h_Z that a population carries. */
export type Predictor = 'tracer' | 'tracerSnapshot' | 'jeansSigma' | 'jeans' | 'sheet'

/** sigma_z^2 ((km/s)^2) and the tilt term t ((km/s)^2/kpc) at the median |z| (kpc) of each slice. */
export interface JeansProfile {
  readonly z: readonly number[]
  readonly sigma2: readonly number[]
  readonly tilt: readonly number[]
}

/** One mono-age population of the annulus. */
export interface AgeBin {
  readonly ageLo: number
  readonly ageHi: number
  readonly ageMedian: number
  readonly sigmaZ: Estimate
  readonly jrMedian: Estimate
  /** h_Z in the self-consistent window (kpc). */
  readonly measured: Estimate
  /** h_Z in Palicio's fixed window (kpc). */
  readonly fixed: number
  readonly sheet: number
  readonly tracer: number
  readonly tracerSnapshot: number
  /** Each predicted h_Z over the measured h_Z, with the error of one bootstrap of both. */
  readonly ratios: Readonly<Record<Predictor, Estimate>>
  /** The slices of the vertical Jeans equation. */
  readonly jeans: JeansProfile
}

/** Age exponents over one range of ages, and the trends of h_pred / h_Z. */
export interface AgeRange {
  readonly lo: number
  readonly hi: number
  readonly exponents: Readonly<Record<string, Estimate>>
  readonly trends: Readonly<Record<Predictor, Trend>>
  readonly eta: number
}

/** The heights of one radial range of the whole disk (kpc). */
export interface Ring {
  readonly rLo: number
  readonly rHi: number
  readonly palicio: number
  readonly scaled: number
  readonly sheet: number
  readonly tracer: number
}

/** The snapshot minus the fitted potential ((km/s)^2) at the median radius of the annulus, per |z| slice. */
export interface Residual {
  readonly z: readonly number[]
  readonly residual: readonly number[]
}

export interface HaloRecord {
  readonly label: string
  readonly zLim: number
  readonly lKpc: number
  readonly bins: readonly AgeBin[]
  readonly ranges: readonly AgeRange[]
  readonly whole: Ring
  readonly residual: Residual
  /** The whole-disk h_Z predicted as a sum of isothermal tracers. */
  readonly tracerSum: TracerSum
  /** The mean vertical motions of every population of the annulus. */
  readonly motions: readonly MotionBin[]
}

/** One age bin of the mock: h_Z and J_R,med from the analysis and from the truth. */
export interface MockBin {
  readonly ageLo: number
  readonly ageHi: number
  readonly hz: number
  readonly hzTrue: number
  readonly jrMedian: number
  readonly jrMedianTrue: number
}

/** The analysis of a mock galaxy of known potential against its truth. */
export interface MockCheck {
  readonly spinErrorDeg: number
  readonly vCirc: number
  readonly vCircTrue: number
  /** The largest |fitted / true - 1| of Delta Phi at the median radius, over 0.05 < z <= 1.5 kpc. */
  readonly deltaPhiError: number
  readonly bins: readonly MockBin[]
}

export interface Spread {
  readonly median: number
  readonly lo: number
  readonly hi: number
}

/** One injected profile: its family, its reference scale (kpc) and the ratios to it. */
export interface InjectionPoint {
  readonly family: string
  readonly reference: number
  readonly fixed: Spread
  readonly fixedFailed: number
  readonly scaled: Spread
}

export interface Injection {
  readonly nStars: number
  readonly draws: number
  readonly zLim: number
  readonly k: number
  readonly points: readonly InjectionPoint[]
}

export interface StatusData {
  readonly halos: readonly HaloRecord[]
  readonly mock: MockCheck
  readonly injection: Injection
}

/** The value of every prediction in `value`, under the keys that the records use. */
function byPredictor<T>(
  value: unknown,
  read: (item: unknown, key: string) => T
): Readonly<Record<Predictor, T>> {
  return {
    tracer: read(value, 'tracer'),
    tracerSnapshot: read(value, 'tracer_snapshot'),
    jeansSigma: read(value, 'jeans_sigma'),
    jeans: read(value, 'jeans'),
    sheet: read(value, 'sheet')
  }
}

function parseBin(bin: unknown): AgeBin {
  const kinematics = field(bin, 'kinematics')
  const heights = field(bin, 'heights')
  const predictions = field(bin, 'predictions')
  return {
    ageLo: num(bin, 'age_lo'),
    ageHi: num(bin, 'age_hi'),
    ageMedian: num(bin, 'age_median'),
    sigmaZ: estimate(kinematics, 'sigma_z'),
    jrMedian: estimate(kinematics, 'jr_median'),
    measured: estimate(heights, 'measured'),
    fixed: num(heights, 'fixed'),
    sheet: num(predictions, 'sheet'),
    tracer: num(predictions, 'tracer'),
    tracerSnapshot: num(predictions, 'tracer_snapshot'),
    ratios: byPredictor(field(bin, 'ratios'), estimate),
    jeans: {
      z: numbers(field(bin, 'jeans'), 'z'),
      sigma2: numbers(field(bin, 'jeans'), 'sigma2'),
      tilt: numbers(field(bin, 'jeans'), 'tilt')
    }
  }
}

/** The exponents that the status pages read from every age range. */
const REQUIRED_EXPONENTS = ['a', 'a_tracer', 'a_sheet', 'hz', 'sigma_z'] as const

function parseRange(range: unknown): AgeRange {
  const exponents = field(range, 'exponents')
  if (typeof exponents !== 'object' || exponents === null) {
    throw new RecordError('"exponents" is not an object')
  }
  for (const key of REQUIRED_EXPONENTS) {
    estimate(exponents, key)
  }
  return {
    lo: num(range, 'lo'),
    hi: num(range, 'hi'),
    exponents: Object.fromEntries(
      Object.keys(exponents).map((key) => [key, estimate(exponents, key)])
    ),
    trends: byPredictor(field(range, 'trends'), trend),
    eta: num(range, 'eta')
  }
}

function parseRing(ring: unknown): Ring {
  const heights = field(ring, 'heights')
  return {
    rLo: num(ring, 'r_lo'),
    rHi: num(ring, 'r_hi'),
    palicio: num(heights, 'palicio'),
    scaled: num(heights, 'scaled'),
    sheet: num(heights, 'sheet'),
    tracer: num(heights, 'tracer')
  }
}

/** The records that the data commands wrote for one halo, as JSON. */
export interface HaloRecords {
  readonly analysis: unknown
  readonly tracerSum: unknown
  readonly equilibrium: unknown
}

/** Check and read the records that `analyze`, `tracer-sum` and `equilibrium` wrote for one halo. */
export function parseHaloRecord(label: string, records: HaloRecords): HaloRecord {
  const record = records.analysis
  const monoAge = field(record, 'mono_age')
  const window = field(field(monoAge, 'method'), 'window')
  return {
    label,
    zLim: num(window, 'z_lim'),
    lKpc: num(window, 'l_kpc'),
    bins: list(monoAge, 'bins').map(parseBin),
    ranges: list(monoAge, 'ranges').map(parseRange),
    whole: parseRing(field(field(record, 'whole_disk'), 'whole')),
    residual: {
      z: numbers(field(record, 'residual'), 'z'),
      residual: numbers(field(record, 'residual'), 'residual')
    },
    tracerSum: parseTracerSum(records.tracerSum),
    motions: parseEquilibrium(records.equilibrium)
  }
}

/** Check and read the record that `experiments auriga mock-check` wrote. */
export function parseMockCheck(record: unknown): MockCheck {
  return {
    spinErrorDeg: num(record, 'spin_error_deg'),
    vCirc: num(record, 'v_circ'),
    vCircTrue: num(record, 'v_circ_true'),
    deltaPhiError: num(record, 'delta_phi_error'),
    bins: list(record, 'bins').map((bin) => ({
      ageLo: num(bin, 'age_lo'),
      ageHi: num(bin, 'age_hi'),
      hz: num(bin, 'hz'),
      hzTrue: num(bin, 'hz_true'),
      jrMedian: num(bin, 'jr_median'),
      jrMedianTrue: num(bin, 'jr_median_true')
    }))
  }
}

function spread(value: unknown, key: string): Spread {
  const item = field(value, key)
  return { median: num(item, 'median'), lo: num(item, 'lo'), hi: num(item, 'hi') }
}

/** Check and read the record that `experiments window-injection` wrote. */
export function parseInjection(record: unknown): Injection {
  return {
    nStars: num(record, 'n_stars'),
    draws: num(record, 'draws'),
    zLim: num(record, 'z_lim'),
    k: num(record, 'k'),
    points: list(record, 'points').map((point) => ({
      family: str(point, 'family'),
      reference: num(point, 'reference'),
      fixed: spread(point, 'fixed'),
      fixedFailed: num(point, 'fixed_failed'),
      scaled: spread(point, 'scaled')
    }))
  }
}

/** Fetch and check the records of the seven halos, the mock and the injection test, under the site's `root`. */
export async function loadStatusData(root: string, signal: AbortSignal): Promise<StatusData> {
  const perHalo = (suffix: string) =>
    Promise.all(HALOS.map((h) => fetchJson(`${root}data/auriga/${h.stem}${suffix}.json`, signal)))
  const [analyses, sums, motions, mock, injection] = await Promise.all([
    perHalo(''),
    perHalo('-tracer-sum'),
    perHalo('-equilibrium'),
    fetchJson(`${root}data/auriga/mock-check.json`, signal),
    fetchJson(`${root}data/window-injection.json`, signal)
  ])
  return {
    halos: HALOS.map((h, i) =>
      parseHaloRecord(h.label, {
        analysis: analyses[i],
        tracerSum: sums[i],
        equilibrium: motions[i]
      })
    ),
    mock: parseMockCheck(mock),
    injection: parseInjection(injection)
  }
}
