import MathLabel from '@site/src/components/MathLabel'
import { Pending } from '@site/src/components/status/controls'
import type { Estimate, StatusData } from '@site/src/components/status/data'
import { FAMILIES } from '@site/src/components/status/families'
import { plusMinus } from '@site/src/components/status/format'
import {
  AGE_RANGES,
  ageRange,
  exponentComparisons,
  extent,
  haloSummaries,
  median,
  type Predictor,
  ratios,
  significance,
  wholeDiskRatios,
  windowComparison,
  within
} from '@site/src/components/status/metrics'
import useStatusData from '@site/src/components/status/useStatusData'
import React, { type ReactNode } from 'react'

export function fixed(value: number, digits = 2): string {
  return Number.isFinite(value) ? value.toFixed(digits) : 'none'
}

export function signed(value: number, digits = 2): string {
  return Number.isFinite(value)
    ? `${value >= 0 ? '+' : '-'}${Math.abs(value).toFixed(digits)}`
    : 'none'
}

export interface TableProps {
  /** Headers spanning several columns, as (label, span) pairs over every column; '' leaves a span blank. */
  readonly groups?: readonly (readonly [string, number])[]
  /** The column headers, with inline math between dollar signs. */
  readonly head: readonly string[]
  readonly rows: readonly (readonly string[])[]
}

/** A table of numbers computed from the records. */
export function Table({ groups, head, rows }: TableProps): ReactNode {
  return (
    <div className="status-table">
      <table>
        <thead>
          {groups === undefined ? null : (
            <tr className="status-table__groups">
              {groups.map(([label, span], i) => (
                <th key={i} colSpan={span}>
                  <MathLabel label={label} />
                </th>
              ))}
            </tr>
          )}
          <tr>
            {head.map((cell, i) => (
              <th key={i}>
                <MathLabel label={cell} />
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.join('|')}>
              {row.map((cell, i) => (
                <td key={i}>
                  <MathLabel label={cell} />
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

/** A table drawn from the records once they have loaded. */
function fromRecords(rows: (data: StatusData) => TableProps): () => ReactNode {
  return function RecordTable(): ReactNode {
    const { value, error } = useStatusData()
    return value === null ? <Pending error={error} /> : <Table {...rows(value)} />
  }
}

const PREDICTORS: readonly (readonly [Predictor, string])[] = [
  ['tracer', 'tracer, fitted $\\Phi$'],
  ['tracerSnapshot', 'tracer, snapshot $\\Phi$'],
  ['jeansSigma', 'Jeans, $\\sigma_z(z)$'],
  ['jeans', 'Jeans, $\\sigma_z(z)$ and tilt'],
  ['sheet', 'sheet']
]

/** How many populations each prediction reaches within 5, 10 and 15 percent. */
export const ToleranceTable = fromRecords((data) => ({
  head: ['prediction', '$\\le 5\\%$', '$\\le 10\\%$', '$\\le 15\\%$', 'median', 'range'],
  rows: PREDICTORS.map(([p, label]) => {
    const values = ratios(data, p)
    const [lo, hi] = extent(values)
    return [
      label,
      ...[0.05, 0.1, 0.15].map((t) => `${within(data, p, t)} / ${values.length}`),
      fixed(median(values)),
      `${fixed(lo)} to ${fixed(hi)}`
    ]
  })
}))

/** The populations of every halo, summarized. */
export const HaloTable = fromRecords((data) => ({
  groups: [
    ['', 2],
    ['within 10 percent', 2],
    ['median ratio', 3],
    ['', 1]
  ],
  head: [
    'halo',
    '$Z_\\mathrm{lim}$ (kpc)',
    'tracer',
    'sheet',
    'tracer',
    'snapshot $\\Phi$',
    'sheet',
    '$\\eta$'
  ],
  rows: haloSummaries(data).map((s) => [
    s.halo,
    fixed(s.zLim, 1),
    `${s.tracerWithin10} / ${s.n}`,
    `${s.sheetWithin10} / ${s.n}`,
    fixed(s.tracerMedian),
    fixed(s.snapshotMedian),
    fixed(s.sheetMedian),
    plusMinus(s.eta.value, s.eta.err, 2, false)
  ])
}))

/** The measured and predicted age exponents of a. */
export const ExponentTable = fromRecords((data) => ({
  head: ['halo', '$\\tau$ (Gyr)', 'measured', 'tracer', 'sheet'],
  rows: exponentComparisons(data).map((c) => [
    c.halo,
    `${c.lo}-${c.hi}`,
    `$${signed(c.measured, 3)} \\pm ${fixed(c.measuredErr, 3)}$`,
    `$${signed(c.tracer, 3)} \\pm ${fixed(c.tracerErr, 3)}$`,
    `$${signed(c.sheet, 3)} \\pm ${fixed(c.sheetErr, 3)}$`
  ])
}))

/** The trends of h_pred / h_Z of the tracer and of the sheet, in standard errors from 0, with chi^2 per degree of freedom. */
export const TrendTable = fromRecords((data) => ({
  groups: [
    ['', 2],
    ['tracer', 3],
    ['sheet', 3]
  ],
  head: [
    'halo',
    '$\\tau$ (Gyr)',
    'trend',
    '$/\\sigma$',
    '$\\chi^2_\\nu$',
    'trend',
    '$/\\sigma$',
    '$\\chi^2_\\nu$'
  ],
  rows: exponentComparisons(data).map((c) => [
    c.halo,
    `${c.lo}-${c.hi}`,
    ...[c.tracerTrend, c.sheetTrend].flatMap((t) => [
      `$${signed(t.value, 3)} \\pm ${fixed(t.err, 3)}$`,
      `$${signed(significance(t), 1)}$`,
      fixed(t.chi2Dof, 1)
    ])
  ])
}))

/** The whole-disk heights over Palicio's radial range, over the self-consistent h_Z. */
export const WholeDiskTable = fromRecords((data) => ({
  head: [
    'halo',
    '$Z_\\mathrm{lim}$ (kpc)',
    '$L$ (kpc)',
    '$h_Z$ (kpc)',
    'Palicio window',
    'sheet',
    'tracer'
  ],
  rows: wholeDiskRatios(data).map((r) => [
    r.halo,
    fixed(r.zLim, 1),
    fixed(r.lKpc, 0),
    fixed(r.scaled),
    fixed(r.palicio),
    fixed(r.sheet),
    fixed(r.tracer)
  ])
}))

function rangeLabel(lo: number, hi: number): string {
  return Number.isFinite(hi) ? `${lo} to ${hi}` : `above ${lo}`
}

/** The median fixed-window ratio of the halos and of each injected family, in ranges of h_Z / Z_lim. */
export const WindowTable = fromRecords((data) => ({
  head: [
    '$h_Z/Z_\\mathrm{lim}$',
    'populations',
    'no solution',
    'Auriga',
    ...FAMILIES.map(([, label]) => label)
  ],
  rows: windowComparison(
    data,
    FAMILIES.map(([name]) => name)
  ).map((row) => [
    rangeLabel(row.lo, row.hi),
    String(row.n),
    String(row.failed),
    fixed(row.halos),
    ...FAMILIES.map(([name]) => fixed(row.curves[name]))
  ])
}))

/** The trends of h_pred / h_Z of the Jeans diagnostics, without and with the tilt term. */
export const JeansTrendTable = fromRecords((data) => ({
  groups: [
    ['', 2],
    ['Jeans, $\\sigma_z(z)$', 2],
    ['Jeans, $\\sigma_z(z)$ and tilt', 2]
  ],
  head: ['halo', '$\\tau$ (Gyr)', 'trend', '$/\\sigma$', 'trend', '$/\\sigma$'],
  rows: data.halos.flatMap((halo) =>
    AGE_RANGES.flatMap(([lo, hi]) => {
      const range = ageRange(halo, lo, hi)
      if (range === undefined) {
        return []
      }
      return [
        [
          halo.label,
          `${lo}-${hi}`,
          ...[range.trends.jeansSigma, range.trends.jeans].flatMap((t) => [
            `$${signed(t.value, 3)} \\pm ${fixed(t.err, 3)}$`,
            `$${signed(significance(t), 1)}$`
          ])
        ]
      ]
    })
  )
}))

/** The whole-disk h_Z of every halo and its predictions as a sum of tracers, over the measured h_Z. */
export const TracerSumTable = fromRecords((data) => ({
  groups: [
    ['', 3],
    ["Palicio's window", 3],
    ['self-consistent window', 3]
  ],
  head: [
    'halo',
    '$R$ (kpc)',
    'unmodeled (%)',
    '$h_Z$ (kpc)',
    'one per ring',
    'per ring and age',
    '$h_Z$ (kpc)',
    'one per ring',
    'per ring and age'
  ],
  rows: data.halos.map((halo) => {
    const whole = halo.tracerSum.whole
    const ratioCell = (e: Estimate) => `$${fixed(e.value, 3)} \\pm ${fixed(e.err, 3)}$`
    return [
      halo.label,
      `${fixed(whole.rLo, 0)}-${fixed(whole.rHi, 0)}`,
      fixed(100 * whole.unmodeledMixture, 1),
      fixed(whole.measured.palicio.value),
      ratioCell(whole.ratioSingle.palicio),
      ratioCell(whole.ratioMixture.palicio),
      fixed(whole.measured.scaled.value),
      ratioCell(whole.ratioSingle.scaled),
      ratioCell(whole.ratioMixture.scaled)
    ]
  })
}))

/** The range of the mean motions of the populations of every halo older than 5 Gyr. */
export const MotionTable = fromRecords((data) => ({
  groups: [
    ['', 1],
    ['populations older than 5 Gyr', 4]
  ],
  head: ['halo', 'thickening (Gyr$^{-1}$)', 'bending (km s$^{-1}$)', 'asymmetry', 'warp (kpc)'],
  rows: data.halos.map((halo) => {
    const old = halo.motions.filter((bin) => bin.ageLo >= 5)
    const span = (values: readonly number[], digits: number) => {
      const [lo, hi] = extent(values)
      return `$${signed(lo, digits)}$ to $${signed(hi, digits)}$`
    }
    return [
      halo.label,
      span(
        old.map((b) => b.thickening.value),
        2
      ),
      span(
        old.map((b) => b.bending.value),
        2
      ),
      span(
        old.map((b) => b.asymmetry.value),
        3
      ),
      `${fixed(Math.min(...old.map((b) => b.warp.value)), 3)} to ${fixed(Math.max(...old.map((b) => b.warp.value)), 3)}`
    ]
  })
}))
