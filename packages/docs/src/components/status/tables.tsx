import MathLabel from '@site/src/components/MathLabel'
import { Pending } from '@site/src/components/status/controls'
import type { StatusData } from '@site/src/components/status/data'
import { FAMILIES } from '@site/src/components/status/families'
import {
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
    fixed(s.eta)
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
