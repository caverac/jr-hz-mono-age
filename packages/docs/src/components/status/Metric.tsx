import type { StatusData } from '@site/src/components/status/data'
import {
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
  wholeDiskRatios,
  windowComparison,
  within,
  worstFailure
} from '@site/src/components/status/metrics'
import useStatusData from '@site/src/components/status/useStatusData'
import React, { type ReactNode } from 'react'

function fixed(value: number, digits = 2): string {
  return value.toFixed(digits)
}

function between([lo, hi]: readonly [number, number], digits = 2): string {
  return `${fixed(lo, digits)} to ${fixed(hi, digits)}`
}

function signed(value: number, digits = 2): string {
  return `${value >= 0 ? '+' : '-'}${Math.abs(value).toFixed(digits)}`
}

function percent(value: number, digits = 1): string {
  return `${(100 * value).toFixed(digits)} percent`
}

/** The comparisons of the exponents over the range of ages that starts at `lo` Gyr. */
function trendsFrom(d: StatusData, lo: number) {
  return exponentComparisons(d).filter((c) => c.lo === lo)
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
          `${c.halo} over ${c.lo}-${c.hi} Gyr (${signed(c.tracerTrend.value, 3)} +/- ${fixed(c.tracerTrend.err, 3)}, ${signed(significance(c.tracerTrend), 1)} standard errors)`
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
  injectionStars: (d: StatusData) => d.injection.nStars.toLocaleString('en-US'),
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
  return <span className="metric">{value === null ? '(loading)' : METRICS[name](value)}</span>
}
