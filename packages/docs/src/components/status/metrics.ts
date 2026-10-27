/**
 * Every number that the status pages state, computed from the records (`data.ts`).
 * The pages never type a result by hand: prose, tables and charts call these functions.
 */

import type {
  AgeBin,
  HaloRecord,
  Injection,
  InjectionPoint,
  MockCheck,
  Predictor,
  StatusData,
  Trend
} from '@site/src/components/status/data'

export type { Predictor } from '@site/src/components/status/data'

/** The age ranges of the exponents, as the records write them. */
export const AGE_RANGES: readonly (readonly [number, number])[] = [
  [0, 10],
  [2, 10]
]

/** Every population of every halo, with its halo. */
export function populations(data: StatusData): readonly (AgeBin & { readonly halo: string })[] {
  return data.halos.flatMap((halo) => halo.bins.map((bin) => ({ ...bin, halo: halo.label })))
}

/** The ratio of the prediction to the measured h_Z of a population. */
export function ratio(bin: AgeBin, predictor: Predictor): number {
  return bin.ratios[predictor].value
}

/** The ratios of the prediction to the measured h_Z over every population with a measured h_Z. */
export function ratios(data: StatusData, predictor: Predictor): readonly number[] {
  return populations(data)
    .map((bin) => ratio(bin, predictor))
    .filter(Number.isFinite)
}

/** How many populations the prediction reaches within the relative `tolerance`. */
export function within(data: StatusData, predictor: Predictor, tolerance: number): number {
  return ratios(data, predictor).filter((r) => Math.abs(r - 1) <= tolerance).length
}

export function median(values: readonly number[]): number {
  const sorted = [...values].filter(Number.isFinite).sort((a, b) => a - b)
  if (sorted.length === 0) {
    return Number.NaN
  }
  const mid = Math.floor(sorted.length / 2)
  return sorted.length % 2 === 1 ? sorted[mid] : 0.5 * (sorted[mid - 1] + sorted[mid])
}

export function extent(values: readonly number[]): readonly [number, number] {
  const finite = values.filter(Number.isFinite)
  return [Math.min(...finite), Math.max(...finite)]
}

/** The range of the halo `halo` over the ages [lo, hi], if the record holds it. */
export function ageRange(halo: HaloRecord, lo: number, hi: number) {
  return halo.ranges.find((r) => r.lo === lo && r.hi === hi)
}

/** The measured and the predicted age exponents of a in one halo and range of ages, and the trends that test them. */
export interface ExponentComparison {
  readonly halo: string
  readonly lo: number
  readonly hi: number
  readonly measured: number
  readonly measuredErr: number
  readonly tracer: number
  readonly tracerErr: number
  readonly sheet: number
  readonly sheetErr: number
  readonly tracerTrend: Trend
  readonly sheetTrend: Trend
}

/** A trend in units of its error: how many standard errors it lies from 0. */
export function significance(trend: Trend): number {
  return trend.value / trend.err
}

/** The comparisons of every halo over every range of `AGE_RANGES`. */
export function exponentComparisons(data: StatusData): readonly ExponentComparison[] {
  return data.halos.flatMap((halo) =>
    AGE_RANGES.flatMap(([lo, hi]) => {
      const range = ageRange(halo, lo, hi)
      if (range === undefined) {
        return []
      }
      const { a, a_tracer: tracer, a_sheet: sheet } = range.exponents
      return [
        {
          halo: halo.label,
          lo,
          hi,
          measured: a.value,
          measuredErr: a.err,
          tracer: tracer.value,
          tracerErr: tracer.err,
          sheet: sheet.value,
          sheetErr: sheet.err,
          tracerTrend: range.trends.tracer,
          sheetTrend: range.trends.sheet
        }
      ]
    })
  )
}

/** The number of ways to choose `k` of `n`. */
function binomial(n: number, k: number): number {
  let result = 1
  for (let i = 1; i <= k; i += 1) {
    result = (result * (n - k + i)) / i
  }
  return result
}

/**
 * The sign test: the two-sided probability of a split of positive and negative `values` at
 * least as unequal as theirs, if each nonzero value were positive or negative with
 * probability 1/2, independently of the others.
 */
export function signTest(values: readonly number[]): number {
  const positive = values.filter((v) => v > 0).length
  const negative = values.filter((v) => v < 0).length
  const n = positive + negative
  let tail = 0
  for (let i = 0; i <= Math.min(positive, negative); i += 1) {
    tail += binomial(n, i)
  }
  return Math.min(1, (2 * tail) / 2 ** n)
}

/** The ratio of eta, d ln h_Z / d ln sigma_z, over 0-10 Gyr, in every halo. */
export function etas(data: StatusData): readonly number[] {
  return data.halos.map((halo) => ageRange(halo, 0, 10)?.eta ?? Number.NaN)
}

/**
 * How much the measured age exponent of a changes when the bins of 0-2 Gyr are left out,
 * in every halo: |beta_a(0-10 Gyr) - beta_a(2-10 Gyr)|. A power law in age would not
 * change.
 */
export function exponentShifts(data: StatusData): readonly number[] {
  return data.halos.map((halo) => {
    const all = ageRange(halo, 0, 10)?.exponents.a.value ?? Number.NaN
    const old = ageRange(halo, 2, 10)?.exponents.a.value ?? Number.NaN
    return Math.abs(all - old)
  })
}

/** The record of the halo `label`; an error if the records do not hold it. */
export function haloRecord(data: StatusData, label: string): HaloRecord {
  const found = data.halos.find((record) => record.label === label)
  if (found === undefined) {
    throw new Error(`no record of ${label}`)
  }
  return found
}

/** The tracer in the snapshot's potential over the tracer in the fitted potential, in every bin of a halo. */
export function snapshotRatios(record: HaloRecord): readonly number[] {
  return record.bins.map((bin) => bin.tracerSnapshot / bin.tracer)
}

/** The snapshot minus the fitted Delta Phi ((km/s)^2) at height `z` (kpc), relative to the slice nearest the plane. */
export function residualAt(record: HaloRecord, z: number): number {
  const { z: heights, residual } = record.residual
  for (let i = 1; i < heights.length; i += 1) {
    if (heights[i - 1] <= z && z <= heights[i]) {
      const t = (z - heights[i - 1]) / (heights[i] - heights[i - 1])
      return residual[i - 1] + t * (residual[i] - residual[i - 1]) - residual[0]
    }
  }
  return Number.NaN
}

/** One row of the summary of a halo. */
export interface HaloSummary {
  readonly halo: string
  readonly zLim: number
  readonly n: number
  readonly tracerWithin10: number
  readonly sheetWithin10: number
  readonly tracerMedian: number
  readonly snapshotMedian: number
  readonly sheetMedian: number
  readonly eta: number
}

/** The populations of every halo, summarized. */
export function haloSummaries(data: StatusData): readonly HaloSummary[] {
  return data.halos.map((record) => {
    const of = (predictor: Predictor) =>
      record.bins.map((bin) => ratio(bin, predictor)).filter(Number.isFinite)
    const inside = (values: readonly number[]) =>
      values.filter((r) => Math.abs(r - 1) <= 0.1).length
    return {
      halo: record.label,
      zLim: record.zLim,
      n: of('tracer').length,
      tracerWithin10: inside(of('tracer')),
      sheetWithin10: inside(of('sheet')),
      tracerMedian: median(of('tracer')),
      snapshotMedian: median(of('tracerSnapshot')),
      sheetMedian: median(of('sheet')),
      eta: ageRange(record, 0, 10)?.eta ?? Number.NaN
    }
  })
}

/** The largest |analysis / truth - 1| of h_Z and of J_R,med over the age bins of the mock. */
export function mockErrors(mock: MockCheck): { readonly hz: number; readonly jr: number } {
  return {
    hz: Math.max(...mock.bins.map((b) => Math.abs(b.hz / b.hzTrue - 1))),
    jr: Math.max(...mock.bins.map((b) => Math.abs(b.jrMedian / b.jrMedianTrue - 1)))
  }
}

/** The whole-disk heights over Palicio's radial range, as ratios to the self-consistent h_Z. */
export function wholeDiskRatios(data: StatusData) {
  return data.halos.map((halo) => ({
    halo: halo.label,
    zLim: halo.zLim,
    lKpc: halo.lKpc,
    scaled: halo.whole.scaled,
    palicio: halo.whole.palicio / halo.whole.scaled,
    sheet: halo.whole.sheet / halo.whole.scaled,
    tracer: halo.whole.tracer / halo.whole.scaled
  }))
}

/** The injected points of one family, sorted by reference / Z_lim. */
export function family(injection: Injection, name: string): readonly InjectionPoint[] {
  return injection.points.filter((p) => p.family === name).sort((a, b) => a.reference - b.reference)
}

/** The median fixed-window ratio of `name` at reference / Z_lim = `x`, interpolated in log-log; NaN outside. */
export function curveAt(injection: Injection, name: string, x: number): number {
  const points = family(injection, name)
    .map((p) => [p.reference / injection.zLim, p.fixed.median] as const)
    .filter(([, y]) => Number.isFinite(y))
  for (let i = 1; i < points.length; i += 1) {
    const [x0, y0] = points[i - 1]
    const [x1, y1] = points[i]
    if (x0 <= x && x <= x1) {
      const t = (Math.log(x) - Math.log(x0)) / (Math.log(x1) - Math.log(x0))
      return Math.exp(Math.log(y0) + t * (Math.log(y1) - Math.log(y0)))
    }
  }
  return Number.NaN
}

/** Every population as a point of the window test: h_Z / Z_lim and fixed-window / self-consistent h_Z. */
export function windowPoints(data: StatusData) {
  return data.halos.flatMap((halo) =>
    halo.bins.map((bin) => ({
      halo: halo.label,
      ageLo: bin.ageLo,
      ageHi: bin.ageHi,
      x: bin.measured.value / halo.zLim,
      y: bin.fixed / bin.measured.value
    }))
  )
}

/** The ranges of h_Z / Z_lim over which the halos are compared with the injected curves. */
export const WINDOW_RANGES: readonly (readonly [number, number])[] = [
  [0, 0.5],
  [0.5, 1],
  [1, 2],
  [2, Number.POSITIVE_INFINITY]
]

/** The median of the halos and of the curves of `families` over each range of `WINDOW_RANGES`. */
export function windowComparison(data: StatusData, families: readonly string[]) {
  const points = windowPoints(data)
  return WINDOW_RANGES.map(([lo, hi]) => {
    const inside = points.filter((p) => p.x >= lo && p.x < hi)
    const measured = inside.filter((p) => Number.isFinite(p.y))
    return {
      lo,
      hi,
      n: inside.length,
      failed: inside.length - measured.length,
      halos: median(measured.map((p) => p.y)),
      curves: Object.fromEntries(
        families.map((name) => [
          name,
          median(measured.map((p) => curveAt(data.injection, name, p.x)))
        ])
      )
    }
  })
}

/** The largest |ratio - 1| of the self-consistent window over every injected profile. */
export function selfConsistentBias(injection: Injection): number {
  return Math.max(...injection.points.map((p) => Math.abs(p.scaled.median - 1)))
}

/** The injected point of `name` whose reference / Z_lim is nearest to `x`, in log. */
export function nearest(injection: Injection, name: string, x: number): InjectionPoint {
  const distance = (p: InjectionPoint) => Math.abs(Math.log(p.reference / injection.zLim / x))
  return [...family(injection, name)].sort((a, b) => distance(a) - distance(b))[0]
}

/** The largest fraction of draws whose fixed-window fit has no solution, and its family. */
export function worstFailure(injection: Injection): InjectionPoint {
  return [...injection.points].sort((a, b) => b.fixedFailed - a.fixedFailed)[0]
}

/** The largest |ratio - 1| of the fixed window for exponentials with reference / Z_lim up to `upTo`. */
export function exponentialFixedBias(injection: Injection, upTo: number): number {
  return Math.max(
    ...family(injection, 'exponential')
      .filter((p) => p.reference / injection.zLim <= upTo)
      .map((p) => Math.abs(p.fixed.median - 1))
  )
}
