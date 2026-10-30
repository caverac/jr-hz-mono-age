/**
 * The numbers of the status pages that the records of the seven halos, the mock and the
 * injection test give, as inline math. They are plain functions of the records, so that the
 * pages and the paper (`scripts/paper-numbers.ts`) quote the same numbers.
 */

import type {
  AgeBin,
  CheckRange,
  CheckSet,
  HaloRecord,
  Predictor,
  StatusData,
  Trend
} from '@site/src/components/status/data'
import type { MotionBin } from '@site/src/components/status/equilibrium'
import {
  between,
  count,
  fixed,
  percent,
  plusMinus,
  signed
} from '@site/src/components/status/format'
import {
  ageRange,
  andList,
  etas,
  exponentComparisons,
  exponentialFixedBias,
  exponentShifts,
  extent,
  family,
  haloRecord,
  median,
  mockErrors,
  nearest,
  populations,
  ratio,
  ratios,
  residualAt,
  selfConsistentBias,
  significance,
  signTest,
  snapshotRatios,
  trendText,
  wholeDiskRatios,
  windowComparison,
  within,
  worstFailure
} from '@site/src/components/status/metrics'
import * as d3 from 'd3'

/** The comparisons of the exponents over the range of ages that starts at `lo` Gyr. */
function trendsFrom(d: StatusData, lo: number) {
  return exponentComparisons(d).filter((c) => c.lo === lo)
}

/** The trend of the prediction `predictor` of every halo over the range of ages from `lo` to 10 Gyr. */
function trendsOf(d: StatusData, predictor: Predictor, lo: number): readonly Trend[] {
  return d.halos.flatMap((h) => {
    const range = ageRange(h, lo, 10)
    return range === undefined ? [] : [range.trends[predictor]]
  })
}

/** The slices with lo <= |z| < hi of the Au19 populations older than 5 Gyr: sigma_z (km/s) and the tilt term. */
function oldSlices(
  d: StatusData,
  lo: number,
  hi: number
): readonly { sigmaZ: number; tilt: number }[] {
  return haloRecord(d, 'Au19')
    .bins.filter((bin) => bin.ageLo >= 5)
    .flatMap((bin) =>
      bin.jeans.z.flatMap((z, i) =>
        z >= lo && z < hi
          ? [{ sigmaZ: Math.sqrt(bin.jeans.sigma2[i]), tilt: bin.jeans.tilt[i] }]
          : []
      )
    )
}

/** The mean motions of the populations of the halo `label` with ages in [lo, hi) Gyr. */
function motionsOf(d: StatusData, label: string, lo: number, hi: number): readonly MotionBin[] {
  return haloRecord(d, label).motions.filter((bin) => bin.ageLo >= lo && bin.ageHi <= hi)
}

/** The mean motions of the populations of every halo with ages in [lo, hi) Gyr. */
function allMotions(d: StatusData, lo: number, hi: number): readonly MotionBin[] {
  return d.halos.flatMap((h) => motionsOf(d, h.label, lo, hi))
}

/** The trend of the prediction `predictor` of Au19 over 0-10 Gyr. */
function au19Trend(d: StatusData, predictor: Predictor): string {
  const range = ageRange(haloRecord(d, 'Au19'), 0, 10)
  if (range === undefined) {
    throw new Error('Au19 has no range of ages over 0-10 Gyr')
  }
  return trendText(range.trends[predictor])
}

/** The sum of tracers over the measured h_Z, in the self-consistent window, in every radial range of every halo. */
function ringMixtureRatios(d: StatusData): readonly number[] {
  return d.halos.flatMap((h) => h.tracerSum.rings.map((r) => r.ratioMixture.scaled.value))
}

/** The comparison of the halos with the injected curves over the range of h_Z / Z_lim that starts at `lo`. */
function windowRow(d: StatusData, lo: number) {
  const row = windowComparison(d, ['sech2', 'exponential']).find((r) => r.lo === lo)
  if (row === undefined) {
    throw new Error(`no range of h_Z / Z_lim starts at ${lo}`)
  }
  return row
}

/** Populations hotter than this (km/s) reach the heights of a spheroid rather than a disk. */
const SPHEROID_SIGMA_Z = 60
/** Populations thinner than this (kpc) sit where the potential is close to harmonic. */
const COLD_HZ = 1
/** Populations thicker than these (kpc) reach where it is not. */
const HOT_HZ = 1.5
const HOTTEST_HZ = 2.5

/** The ratios of `predictor` over the populations that `keep` selects. */
function ratiosWhere(
  d: StatusData,
  predictor: Predictor,
  keep: (bin: AgeBin) => boolean
): readonly number[] {
  return populations(d)
    .filter(keep)
    .map((bin) => ratio(bin, predictor))
    .filter(Number.isFinite)
}

/** How many of `values` lie within 10 percent of 1, out of how many. */
function within10Of(values: readonly number[]): string {
  return `${values.filter((r) => Math.abs(r - 1) <= 0.1).length} of ${values.length}`
}

/** The root mean square of `values` - 1. */
function rms(values: readonly number[]): number {
  return Math.sqrt(values.reduce((sum, r) => sum + (r - 1) ** 2, 0) / values.length)
}

/** The point of the tracer curve of `halo` nearest `sigmaZ` (km/s). */
function curveAt(
  halo: HaloRecord,
  sigmaZ: number
): { readonly sigmaZ: number; readonly eta: number } {
  const curve = halo.tracerCurve
  const i = d3.minIndex(curve.sigmaZ, (s) => Math.abs(Math.log(s / sigmaZ)))
  return { sigmaZ: curve.sigmaZ[i], eta: curve.eta[i] }
}

/** The smallest sigma_z (km/s) of the curve at which the window finds no scale; NaN if it always does. */
function firstWithoutScale(halo: HaloRecord): number {
  const curve = halo.tracerCurve
  return curve.sigmaZ.find((_, i) => !Number.isFinite(curve.hz[i])) ?? Number.NaN
}

/** ln h_Z of the tracer curve of `halo` at `sigmaZ` (km/s), interpolated linearly in ln sigma_z. */
function curveLogHeight(halo: HaloRecord, sigmaZ: number): number {
  const { sigmaZ: grid, hz } = halo.tracerCurve
  const i = Math.min(Math.max(d3.bisectRight(grid, sigmaZ), 1), grid.length - 1)
  const t = Math.log(sigmaZ / grid[i - 1]) / Math.log(grid[i] / grid[i - 1])
  return (1 - t) * Math.log(hz[i - 1]) + t * Math.log(hz[i])
}

/**
 * The mean local slope of the tracer curve of `halo` over the sigma_z of its populations of
 * 0-10 Gyr: the slope of the chord of ln h_Z against ln sigma_z between their extremes.
 */
function chord(halo: HaloRecord): number {
  const [lo, hi] = extent(halo.bins.filter((b) => b.ageHi <= 10).map((b) => b.sigmaZ.value))
  return (curveLogHeight(halo, hi) - curveLogHeight(halo, lo)) / Math.log(hi / lo)
}

/** The trend of a range of ages that a halo does not have. */
const NO_TREND: Trend = { value: Number.NaN, err: Number.NaN, chi2Dof: Number.NaN }

/** eta over 0-10 Gyr of `halo`. */
function eta010(halo: HaloRecord): Trend {
  return ageRange(halo, 0, 10)?.eta ?? NO_TREND
}

/** The populations that have a thin-layer ratio, those whose orbits stay inside the grid of heights. */
function withCoeval(d: StatusData): readonly AgeBin[] {
  return populations(d).filter((b) => Number.isFinite(b.coeval.ratio.value))
}

/** The values of `values` that are finite. */
function finite(values: readonly number[]): readonly number[] {
  return values.filter(Number.isFinite)
}

/** The range of the check `set` over [lo, 10] Gyr. */
function checkRange(set: CheckSet, lo: number): CheckRange {
  const found = set.ranges.find((r) => r.lo === lo && r.hi === 10)
  if (found === undefined) {
    throw new Error(`no range of a check over ${lo}-10 Gyr`)
  }
  return found
}

/** The largest |trend| / error of the tracer ratio of the check `of` over [lo, 10] Gyr across the halos. */
function maxSignificance(d: StatusData, of: (h: HaloRecord) => CheckSet, lo: number): number {
  return Math.max(
    ...d.halos.map((h) => {
      const t = checkRange(of(h), lo).tracer
      return Math.abs(t.value) / t.err
    })
  )
}

/** How many of the halos have a negative trend `of`, out of how many. */
function negativeOf(d: StatusData, of: (h: HaloRecord) => Trend): string {
  return `${d.halos.filter((h) => of(h).value < 0).length} of ${d.halos.length}`
}

const isCold = (bin: AgeBin) => bin.measured.value < COLD_HZ
const isHot = (bin: AgeBin) => bin.measured.value > HOT_HZ
const isHottest = (bin: AgeBin) => bin.measured.value > HOTTEST_HZ
const isDisk = (bin: AgeBin) => bin.sigmaZ.value <= SPHEROID_SIGMA_Z

/** The numbers that the prose of the status pages and the paper quote, by name. */
export const STATUS_METRICS = {
  populations: (d: StatusData) => String(populations(d).length),
  tracerWithin5: (d: StatusData) => String(within(d, 'tracer', 0.05)),
  tracerWithin10: (d: StatusData) => String(within(d, 'tracer', 0.1)),
  tracerWithin15: (d: StatusData) => String(within(d, 'tracer', 0.15)),
  sheetWithin10: (d: StatusData) => String(within(d, 'sheet', 0.1)),
  tracerRange: (d: StatusData) => between(extent(ratios(d, 'tracer'))),
  tracerMedian: (d: StatusData) => fixed(median(ratios(d, 'tracer'))),
  sheetRange: (d: StatusData) => between(extent(ratios(d, 'sheet'))),
  trendMax0: (d: StatusData) =>
    fixed(Math.max(...trendsFrom(d, 0).map((c) => Math.abs(significance(c.tracerTrend)))), 1),
  trendRange0: (d: StatusData) =>
    between(extent(trendsFrom(d, 0).map((c) => c.tracerTrend.value)), 3),
  trendOutside2: (d: StatusData) => {
    const outside = exponentComparisons(d)
      .filter((c) => Math.abs(significance(c.tracerTrend)) > 2)
      .map(
        (c) =>
          `${c.halo} over ${c.lo}-${c.hi} Gyr (${plusMinus(c.tracerTrend.value, c.tracerTrend.err)}, ${signed(significance(c.tracerTrend), 1)} standard errors)`
      )
    return outside.length === 0 ? 'no case' : andList(outside)
  },
  trendNegative2: (d: StatusData) => {
    const values = trendsFrom(d, 2).map((c) => c.tracerTrend.value)
    return `${values.filter((v) => v < 0).length} of the ${values.length}`
  },
  trendMedian2: (d: StatusData) =>
    signed(median(trendsFrom(d, 2).map((c) => c.tracerTrend.value)), 3),
  trendSignTest2: (d: StatusData) =>
    fixed(signTest(trendsFrom(d, 2).map((c) => c.tracerTrend.value)), 3),
  jeansSigmaWithin10: (d: StatusData) => String(within(d, 'jeansSigma', 0.1)),
  jeansWithin10: (d: StatusData) => String(within(d, 'jeans', 0.1)),
  jeansSigmaMedian: (d: StatusData) => fixed(median(ratios(d, 'jeansSigma'))),
  jeansMedian: (d: StatusData) => fixed(median(ratios(d, 'jeans'))),
  jeansSigmaNegative2: (d: StatusData) => {
    const values = trendsOf(d, 'jeansSigma', 2).map((t) => t.value)
    return `${values.filter((v) => v < 0).length} of the ${values.length}`
  },
  jeansSigmaWithin2: (d: StatusData) =>
    String(
      [...trendsOf(d, 'jeansSigma', 0), ...trendsOf(d, 'jeansSigma', 2)].filter(
        (t) => Math.abs(significance(t)) <= 2
      ).length
    ),
  jeansWithin2: (d: StatusData) =>
    String(
      [...trendsOf(d, 'jeans', 0), ...trendsOf(d, 'jeans', 2)].filter(
        (t) => Math.abs(significance(t)) <= 2
      ).length
    ),
  jeansSignTest2: (d: StatusData) =>
    fixed(signTest(trendsOf(d, 'jeans', 2).map((t) => t.value)), 3),
  residualRange: (d: StatusData) => between(extent(d.halos.map((h) => residualAt(h, 1))), 0),
  snapshotRange: (d: StatusData) => between(extent(d.halos.flatMap((h) => snapshotRatios(h)))),
  tracerChi2Range: (d: StatusData) =>
    between(extent(exponentComparisons(d).map((c) => c.tracerTrend.chi2Dof)), 1),
  sheetTrendRange: (d: StatusData) =>
    between(
      extent(
        exponentComparisons(d)
          .filter((c) => c.lo === 0)
          .map((c) => c.sheetTrend.value)
      )
    ),
  sheetTrendMin: (d: StatusData) =>
    fixed(Math.min(...exponentComparisons(d).map((c) => Math.abs(significance(c.sheetTrend)))), 0),
  exponentShift: (d: StatusData) => between(extent(exponentShifts(d))),
  etaRange: (d: StatusData) => between(extent(etas(d))),
  halfMassTracerWithin10: (d: StatusData) =>
    within10Of(populations(d).map((b) => b.halfMassRatios.tracer.value)),
  halfMassJeansWithin10: (d: StatusData) =>
    within10Of(populations(d).map((b) => b.halfMassRatios.jeans.value)),
  halfMassSheetWithin10: (d: StatusData) =>
    within10Of(populations(d).map((b) => b.halfMassRatios.sheet.value)),
  halfMassTracerRange: (d: StatusData) =>
    between(extent(populations(d).map((b) => b.halfMassRatios.tracer.value))),
  halfMassTracerMedian: (d: StatusData) =>
    fixed(median(populations(d).map((b) => b.halfMassRatios.tracer.value))),
  halfMassTracerRms: (d: StatusData) =>
    percent(rms(populations(d).map((b) => b.halfMassRatios.tracer.value))),
  halfMassTrendMax0: (d: StatusData) =>
    fixed(
      Math.max(
        ...d.halos.map((h) => {
          const t = ageRange(h, 0, 10)?.halfMassTrends.tracer
          return t === undefined ? Number.NaN : Math.abs(t.value) / t.err
        })
      ),
      1
    ),
  etaHalfMassRange: (d: StatusData) =>
    between(extent(d.halos.map((h) => ageRange(h, 0, 10)?.etaHalfMass.value ?? Number.NaN))),
  curveEtaAt10: (d: StatusData) => between(extent(d.halos.map((h) => curveAt(h, 10).eta))),
  curveEtaAt40: (d: StatusData) => between(extent(d.halos.map((h) => curveAt(h, 40).eta))),
  curveNoScale: (d: StatusData) => between(extent(d.halos.map((h) => firstWithoutScale(h))), 0),
  diskMassRange: (d: StatusData) =>
    between(extent(d.halos.flatMap((h) => h.checks.disk.bins.map((b) => b.massFraction)))),
  diskWithin10: (d: StatusData) =>
    within10Of(finite(d.halos.flatMap((h) => h.checks.disk.bins.map((b) => b.tracer.value)))),
  diskTrendMax0: (d: StatusData) =>
    fixed(
      maxSignificance(d, (h) => h.checks.disk, 0),
      1
    ),
  diskTrendRange0: (d: StatusData) =>
    between(extent(d.halos.map((h) => checkRange(h.checks.disk, 0).tracer.value)), 3),
  diskTrendNegative2: (d: StatusData) => negativeOf(d, (h) => checkRange(h.checks.disk, 2).tracer),
  fineWithin10: (d: StatusData) =>
    within10Of(finite(d.halos.flatMap((h) => h.checks.fine.bins.map((b) => b.tracer.value)))),
  fineTrendRange2: (d: StatusData) =>
    between(extent(d.halos.map((h) => checkRange(h.checks.fine, 2).tracer.value)), 3),
  fineTrendNegative2: (d: StatusData) => negativeOf(d, (h) => checkRange(h.checks.fine, 2).tracer),
  fineTrendMax2: (d: StatusData) =>
    fixed(
      maxSignificance(d, (h) => h.checks.fine, 2),
      1
    ),
  unclippedShift: (d: StatusData) =>
    `${between(
      extent(
        d.halos.flatMap((h) =>
          h.checks.allStars.bins.map((b) => 100 * (b.tracerUnclipped.value / b.tracer.value - 1))
        )
      ),
      0
    )} percent`,
  unclippedTrendRange2: (d: StatusData) =>
    between(extent(d.halos.map((h) => checkRange(h.checks.allStars, 2).tracerUnclipped.value)), 3),
  unclippedTrendNegative2: (d: StatusData) =>
    negativeOf(d, (h) => checkRange(h.checks.allStars, 2).tracerUnclipped),
  etaErrRange: (d: StatusData) => between(extent(d.halos.map((h) => eta010(h).err))),
  curveEtaMax: (d: StatusData) =>
    between(extent(d.halos.map((h) => Math.max(...finite(h.tracerCurve.eta))))),
  curveNoScaleCount: (d: StatusData) =>
    `${d.halos.filter((h) => Number.isFinite(firstWithoutScale(h))).length} of ${d.halos.length}`,
  noScaleMargin: (d: StatusData) =>
    fixed(
      Math.min(
        ...d.halos.flatMap((h) => {
          const limit = firstWithoutScale(h)
          const hottest = Math.max(...h.bins.map((b) => b.sigmaZ.value))
          return Number.isFinite(limit) ? [limit / hottest] : []
        })
      ),
      1
    ),
  etaChordDiff: (d: StatusData) => between(extent(d.halos.map((h) => eta010(h).value - chord(h)))),
  etaChordWithin2: (d: StatusData) => {
    const close = d.halos.filter((h) => Math.abs(eta010(h).value - chord(h)) <= 2 * eta010(h).err)
    return `${close.length} of ${d.halos.length}`
  },
  au19Eta: (d: StatusData) => {
    const e = eta010(haloRecord(d, 'Au19'))
    return plusMinus(e.value, e.err, 2, false)
  },
  au19Chord: (d: StatusData) => fixed(chord(haloRecord(d, 'Au19'))),
  halfMassTracerAbove1: (d: StatusData) => {
    const values = populations(d).map((b) => b.halfMassRatios.tracer.value)
    return `${values.filter((r) => r > 1).length} of ${values.length}`
  },
  halfMassJeansSigmaWithin10: (d: StatusData) =>
    within10Of(populations(d).map((b) => b.halfMassRatios.jeansSigma.value)),
  halfMassJeansRms: (d: StatusData) =>
    percent(rms(populations(d).map((b) => b.halfMassRatios.jeans.value))),
  halfMassSheetMedian: (d: StatusData) =>
    fixed(median(populations(d).map((b) => b.halfMassRatios.sheet.value))),
  halfMassTrendNegative2: (d: StatusData) =>
    negativeOf(d, (h) => ageRange(h, 2, 10)?.halfMassTrends.tracer ?? NO_TREND),
  halfMassTrendMax2: (d: StatusData) =>
    fixed(
      Math.max(
        ...d.halos.map((h) => {
          const t = ageRange(h, 2, 10)?.halfMassTrends.tracer
          return t === undefined ? Number.NaN : Math.abs(t.value) / t.err
        })
      ),
      1
    ),
  unclippedWithin10: (d: StatusData) =>
    within10Of(
      finite(d.halos.flatMap((h) => h.checks.allStars.bins.map((b) => b.tracerUnclipped.value)))
    ),
  unclippedMedian: (d: StatusData) =>
    fixed(
      median(
        finite(d.halos.flatMap((h) => h.checks.allStars.bins.map((b) => b.tracerUnclipped.value)))
      )
    ),
  coevalWithin10: (d: StatusData) =>
    within10Of(finite(populations(d).map((b) => b.coeval.ratio.value))),
  coevalTracerWithin10: (d: StatusData) =>
    within10Of(withCoeval(d).map((b) => b.ratios.tracer.value)),
  coevalTracerMedian: (d: StatusData) =>
    fixed(median(withCoeval(d).map((b) => b.ratios.tracer.value))),
  coevalCount: (d: StatusData) => {
    const all = populations(d)
    return `${finite(all.map((b) => b.coeval.ratio.value)).length} of ${all.length}`
  },
  coevalMedian: (d: StatusData) => fixed(median(populations(d).map((b) => b.coeval.ratio.value))),
  coevalRms: (d: StatusData) =>
    percent(rms(finite(populations(d).map((b) => b.coeval.ratio.value)))),
  coevalHalfMassMedian: (d: StatusData) =>
    fixed(median(populations(d).map((b) => b.coeval.halfMassRatio.value))),
  coevalTrendRange0: (d: StatusData) =>
    between(extent(d.halos.map((h) => ageRange(h, 0, 10)?.coevalTrends.hz.value ?? Number.NaN)), 3),
  bFitRange: (d: StatusData) => between(extent(populations(d).map((b) => b.coeval.b.value))),
  bFitErrRange: (d: StatusData) => between(extent(populations(d).map((b) => b.coeval.b.err))),
  bIsoRange: (d: StatusData) => between(extent(populations(d).map((b) => b.coeval.bIsothermal))),
  bExcessRange: (d: StatusData) =>
    between(extent(populations(d).map((b) => b.coeval.b.value - b.coeval.bIsothermal))),
  bExcessMedian: (d: StatusData) =>
    fixed(median(populations(d).map((b) => b.coeval.b.value - b.coeval.bIsothermal))),
  bAboveIso: (d: StatusData) => {
    const all = populations(d).filter((b) => Number.isFinite(b.coeval.bIsothermal))
    return `${all.filter((b) => b.coeval.b.value > b.coeval.bIsothermal).length} of ${all.length}`
  },
  bColdRange: (d: StatusData) =>
    between(
      extent(
        populations(d)
          .filter(isCold)
          .map((b) => b.coeval.b.value)
      )
    ),
  bIsoColdRange: (d: StatusData) =>
    between(
      extent(
        populations(d)
          .filter(isCold)
          .map((b) => b.coeval.bIsothermal)
      )
    ),
  betaZRange0: (d: StatusData) =>
    between(extent(d.halos.map((h) => ageRange(h, 0, 10)?.exponents.sigma_z.value ?? Number.NaN))),
  betaHRange0: (d: StatusData) =>
    between(extent(d.halos.map((h) => ageRange(h, 0, 10)?.exponents.hz.value ?? Number.NaN))),
  etaBetaZRange0: (d: StatusData) =>
    between(
      extent(
        d.halos.map((h) => {
          const r = ageRange(h, 0, 10)
          return r === undefined ? Number.NaN : r.eta.value * r.exponents.sigma_z.value
        })
      )
    ),
  twoBetaZRange0: (d: StatusData) =>
    between(
      extent(d.halos.map((h) => 2 * (ageRange(h, 0, 10)?.exponents.sigma_z.value ?? Number.NaN)))
    ),
  etaAllRanges: (d: StatusData) =>
    between(extent(d.halos.flatMap((h) => h.ranges.map((r) => r.eta.value)))),
  tracerRms: (d: StatusData) => percent(rms(ratios(d, 'tracer'))),
  jeansRms: (d: StatusData) => percent(rms(ratios(d, 'jeans'))),
  jeansRange: (d: StatusData) => between(extent(ratios(d, 'jeans'))),
  coldTracer: (d: StatusData) => within10Of(ratiosWhere(d, 'tracer', isCold)),
  coldTracerRms: (d: StatusData) => percent(rms(ratiosWhere(d, 'tracer', isCold))),
  hotTracer: (d: StatusData) => within10Of(ratiosWhere(d, 'tracer', isHot)),
  hotJeans: (d: StatusData) => within10Of(ratiosWhere(d, 'jeans', isHot)),
  hottestTracerRms: (d: StatusData) => percent(rms(ratiosWhere(d, 'tracer', isHottest))),
  hottestJeansRms: (d: StatusData) => percent(rms(ratiosWhere(d, 'jeans', isHottest))),
  diskTracer: (d: StatusData) => within10Of(ratiosWhere(d, 'tracer', isDisk)),
  diskTracerRms: (d: StatusData) => percent(rms(ratiosWhere(d, 'tracer', isDisk))),
  spheroidTracer: (d: StatusData) => within10Of(ratiosWhere(d, 'tracer', (bin) => !isDisk(bin))),
  sigmaZMax: (d: StatusData) => fixed(Math.max(...populations(d).map((b) => b.sigmaZ.value)), 0),
  hzMax: (d: StatusData) => fixed(Math.max(...populations(d).map((b) => b.measured.value)), 1),
  wholeTracer: (d: StatusData) => between(extent(wholeDiskRatios(d).map((r) => r.tracer))),
  wholePalicio: (d: StatusData) => between(extent(wholeDiskRatios(d).map((r) => r.palicio))),
  mockSpin: (d: StatusData) => fixed(d.mock.spinErrorDeg, 3),
  mockVc: (d: StatusData) => percent(Math.abs(d.mock.vCirc / d.mock.vCircTrue - 1)),
  mockPhi: (d: StatusData) => percent(d.mock.deltaPhiError),
  mockHz: (d: StatusData) => percent(mockErrors(d.mock).hz),
  mockJr: (d: StatusData) => percent(mockErrors(d.mock).jr),
  injectionDraws: (d: StatusData) => String(d.injection.draws),
  injectionStars: (d: StatusData) => count(d.injection.nStars),
  selfConsistentBias: (d: StatusData) => percent(selfConsistentBias(d.injection)),
  exponentialFixedBias: (d: StatusData) => percent(exponentialFixedBias(d.injection, 3)),
  exponentialSpreadWide: (d: StatusData) => {
    const widest = family(d.injection, 'exponential').at(-1)
    return widest === undefined ? 'none' : between([widest.fixed.lo, widest.fixed.hi])
  },
  sech2Thin: (d: StatusData) => fixed(nearest(d.injection, 'sech2', 0.2).fixed.median),
  sech2Window: (d: StatusData) => fixed(nearest(d.injection, 'sech2', 1).fixed.median),
  fixedFailedMax: (d: StatusData) => percent(worstFailure(d.injection).fixedFailed, 0),
  windowLowHalos: (d: StatusData) => fixed(windowRow(d, 0).halos),
  windowLowSech2: (d: StatusData) => fixed(windowRow(d, 0).curves.sech2),
  windowLowExponential: (d: StatusData) => fixed(windowRow(d, 0).curves.exponential),
  windowMidHalos: (d: StatusData) => fixed(windowRow(d, 0.5).halos),
  windowMidSech2: (d: StatusData) => fixed(windowRow(d, 0.5).curves.sech2),
  windowMidExponential: (d: StatusData) => fixed(windowRow(d, 0.5).curves.exponential),
  windowHighHalos: (d: StatusData) => fixed(windowRow(d, 1).halos),
  windowHighSech2: (d: StatusData) => fixed(windowRow(d, 1).curves.sech2),
  windowHighExponential: (d: StatusData) => fixed(windowRow(d, 1).curves.exponential),
  zLimRange: (d: StatusData) => between(extent(d.halos.map((h) => h.zLim)), 1),
  au19JeansOld: (d: StatusData) =>
    between(
      extent(
        haloRecord(d, 'Au19')
          .bins.filter((bin) => bin.ageLo >= 5)
          .map((bin) => ratio(bin, 'jeansSigma'))
      )
    ),
  au19JeansTiltOld: (d: StatusData) =>
    between(
      extent(
        haloRecord(d, 'Au19')
          .bins.filter((bin) => bin.ageLo >= 5)
          .map((bin) => ratio(bin, 'jeans'))
      )
    ),
  au19TiltNegative: (d: StatusData) => {
    const tilts = oldSlices(d, 1, 5).map((s) => s.tilt)
    return `${tilts.filter((t) => t < 0).length} of the ${tilts.filter((t) => !Number.isNaN(t)).length}`
  },
  au19SigmaPlane: (d: StatusData) => between(extent(oldSlices(d, 0, 0.2).map((s) => s.sigmaZ)), 0),
  au19SigmaHigh: (d: StatusData) => between(extent(oldSlices(d, 6, 12).map((s) => s.sigmaZ)), 0),
  au19HzOld: (d: StatusData) =>
    between(
      extent(
        haloRecord(d, 'Au19')
          .bins.filter((bin) => bin.ageLo >= 5)
          .map((bin) => bin.measured.value)
      )
    ),
  jeansPositive2: (d: StatusData) => {
    const values = trendsOf(d, 'jeans', 2).map((t) => t.value)
    return `${values.filter((v) => v > 0).length} of the ${values.length}`
  },
  au19JeansSigmaTrend0: (d: StatusData) => au19Trend(d, 'jeansSigma'),
  au19JeansTrend0: (d: StatusData) => au19Trend(d, 'jeans'),
  sumSingleScaled: (d: StatusData) =>
    between(extent(d.halos.map((h) => h.tracerSum.whole.ratioSingle.scaled.value))),
  sumMixtureScaled: (d: StatusData) =>
    between(extent(d.halos.map((h) => h.tracerSum.whole.ratioMixture.scaled.value))),
  sumRingsScaled: (d: StatusData) => between(extent(ringMixtureRatios(d))),
  sumRingsWithin10: (d: StatusData) => {
    const values = ringMixtureRatios(d)
    return `${values.filter((v) => Math.abs(v - 1) <= 0.1).length} of the ${values.length}`
  },
  sumSinglePalicio: (d: StatusData) =>
    between(extent(d.halos.map((h) => h.tracerSum.whole.ratioSingle.palicio.value))),
  sumMixturePalicio: (d: StatusData) =>
    between(extent(d.halos.map((h) => h.tracerSum.whole.ratioMixture.palicio.value))),
  au19ThickeningOld: (d: StatusData) =>
    between(extent(motionsOf(d, 'Au19', 5, 10).map((b) => b.thickening.value)), 2),
  au19AsymmetryOld: (d: StatusData) =>
    between(extent(motionsOf(d, 'Au19', 5, 10).map((b) => b.asymmetry.value)), 3),
  oldThickeningMax: (d: StatusData) =>
    fixed(Math.max(...allMotions(d, 5, 10).map((b) => Math.abs(b.thickening.value))), 1),
  youngThickeningMax: (d: StatusData) =>
    fixed(Math.max(...allMotions(d, 0, 5).map((b) => Math.abs(b.thickening.value))), 1),
  youngBendingMax: (d: StatusData) =>
    fixed(Math.max(...allMotions(d, 0, 5).map((b) => Math.abs(b.bending.value))), 1),
  youngAsymmetryMax: (d: StatusData) =>
    fixed(Math.max(...allMotions(d, 0, 5).map((b) => Math.abs(b.asymmetry.value))), 2),
  sumSingleMedian: (d: StatusData) =>
    fixed(median(d.halos.map((h) => h.tracerSum.whole.ratioSingle.scaled.value)), 3),
  sumMixtureMedian: (d: StatusData) =>
    fixed(median(d.halos.map((h) => h.tracerSum.whole.ratioMixture.scaled.value)), 3),
  sumMixtureCloser: (d: StatusData) => {
    const closer = d.halos.filter((h) => {
      const whole = h.tracerSum.whole
      return (
        Math.abs(whole.ratioMixture.scaled.value - 1) < Math.abs(whole.ratioSingle.scaled.value - 1)
      )
    })
    return `${closer.length} of the ${d.halos.length}`
  },
  sumMixtureShort: (d: StatusData) => {
    const short = d.halos.filter((h) => h.tracerSum.whole.ratioMixture.scaled.value < 1)
    return `${short.length} of the ${d.halos.length}`
  },
  au19ZLim: (d: StatusData) => fixed(haloRecord(d, 'Au19').zLim, 1),
  au19WholePalicio: (d: StatusData) =>
    fixed(haloRecord(d, 'Au19').tracerSum.whole.measured.palicio.value, 1),
  sumUnmodeledMax: (d: StatusData) =>
    percent(Math.max(...d.halos.map((h) => h.tracerSum.whole.unmodeledMixture)), 1),
  au19Old: (d: StatusData) =>
    between(
      extent(
        haloRecord(d, 'Au19')
          .bins.filter((bin) => bin.ageLo >= 5)
          .map((bin) => ratio(bin, 'tracer'))
      )
    ),
  mixtureThin: (d: StatusData) => {
    const thinnest = family(d.injection, 'mixture')[0]
    return thinnest === undefined ? 'none' : fixed(thinnest.fixed.median)
  },
  sheetYoung: (d: StatusData) => between(extent(d.halos.map((h) => ratio(h.bins[0], 'sheet')))),
  sheetOld: (d: StatusData) =>
    between(extent(d.halos.map((h) => ratio(h.bins[h.bins.length - 1], 'sheet'))))
} satisfies Record<string, (d: StatusData) => string>

export type MetricName = keyof typeof STATUS_METRICS
