import MathLabel from '@site/src/components/MathLabel'
import type { Predictor, StatusData, Trend } from '@site/src/components/status/data'
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
import useStatusData from '@site/src/components/status/useStatusData'
import React, { type ReactNode } from 'react'

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

/** The comparison of the halos with the injected curves over the range of h_Z / Z_lim that starts at `lo`. */
function windowRow(d: StatusData, lo: number) {
  const row = windowComparison(d, ['sech2', 'exponential']).find((r) => r.lo === lo)
  if (row === undefined) {
    throw new Error(`no range of h_Z / Z_lim starts at ${lo}`)
  }
  return row
}

/** The numbers that the prose of the status pages quotes, by name. */
const METRICS = {
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
    return outside.length === 0
      ? 'no case'
      : `${outside.slice(0, -1).join(', ')}${outside.length > 1 ? ' and ' : ''}${outside.at(-1)}`
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
  jeansSigmaRange: (d: StatusData) => between(extent(ratios(d, 'jeansSigma'))),
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
  jeansSigmaSignTest2: (d: StatusData) =>
    fixed(signTest(trendsOf(d, 'jeansSigma', 2).map((t) => t.value)), 2),
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
  wholeSheet: (d: StatusData) => between(extent(wholeDiskRatios(d).map((r) => r.sheet))),
  wholeTracer: (d: StatusData) => between(extent(wholeDiskRatios(d).map((r) => r.tracer))),
  wholePalicio: (d: StatusData) => between(extent(wholeDiskRatios(d).map((r) => r.palicio))),
  au23Residual: (d: StatusData) => fixed(residualAt(haloRecord(d, 'Au23'), 1), 0),
  au23Snapshot: (d: StatusData) => between(extent(snapshotRatios(haloRecord(d, 'Au23')))),
  otherResidual: (d: StatusData) =>
    fixed(
      Math.max(...d.halos.filter((h) => h.label !== 'Au23').map((h) => Math.abs(residualAt(h, 1)))),
      0
    ),
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

export type MetricName = keyof typeof METRICS

/**
 * A number of the status pages, computed from the records when the page loads, so that
 * the prose cannot drift from the data.
 */
export default function Metric({ name }: { readonly name: MetricName }): ReactNode {
  const { value, error } = useStatusData()
  if (error !== null) {
    return <span className="metric metric--error">(no data: {error})</span>
  }
  return (
    <span className="metric">
      {value === null ? '(loading)' : <MathLabel label={METRICS[name](value)} />}
    </span>
  )
}
