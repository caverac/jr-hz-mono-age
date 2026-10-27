import { Pending } from '@site/src/components/status/controls'
import {
  JR_RELATION_PATH,
  type JrRelationData,
  parseJrRelation
} from '@site/src/components/status/jrRelation'
import { significance } from '@site/src/components/status/metrics'
import {
  MILKY_WAY_PATH,
  type MilkyWayData,
  parseMilkyWay
} from '@site/src/components/status/milkyWay'
import { fixed, signed, Table, type TableProps } from '@site/src/components/status/tables'
import useRecord from '@site/src/components/status/useRecord'
import React, { type ReactNode } from 'react'

/** A table drawn from the record at `path` once it has loaded. */
function fromRecord<T>(
  path: string,
  parse: (record: unknown) => T,
  rows: (data: T) => TableProps
): () => ReactNode {
  return function RecordTable(): ReactNode {
    const { value, error } = useRecord(path, parse)
    return value === null ? <Pending error={error} /> : <Table {...rows(value)} />
  }
}

const NAMES: Record<string, string> = { lamost: 'LAMOST', apogee: 'APOGEE' }

function estimateCell(value: number, err: number, digits: number): string {
  return `$${fixed(value, digits)} \\pm ${fixed(err, digits)}$`
}

/** Every age bin of both surveys of the Milky Way. */
export const MwBinTable = fromRecord(MILKY_WAY_PATH, parseMilkyWay, (data: MilkyWayData) => ({
  head: [
    'survey',
    '$\\tau$ (Gyr)',
    'median',
    'matches',
    'stars',
    '$\\sigma_z$ (km s$^{-1}$)',
    '$h_Z$ (kpc)',
    'tracer',
    'sheet'
  ],
  rows: data.surveys.flatMap((survey) =>
    survey.bins.map((b) => [
      NAMES[survey.name] ?? survey.name,
      `${b.ageLo}-${b.ageHi}`,
      fixed(b.ageMedian),
      String(b.nMatched),
      String(b.nKin),
      estimateCell(b.sigmaZ.value, b.sigmaZ.err, 1),
      estimateCell(b.measured.value, b.measured.err, 3),
      estimateCell(b.ratioTracer.value, b.ratioTracer.err, 2),
      estimateCell(b.ratioSheet.value, b.ratioSheet.err, 2)
    ])
  )
}))

/** The trends of h_pred / h_Z in both surveys of the Milky Way and against Imig et al. */
export const MwTrendTable = fromRecord(MILKY_WAY_PATH, parseMilkyWay, (data: MilkyWayData) => ({
  groups: [
    ['', 3],
    ['tracer', 3],
    ['sheet', 2]
  ],
  head: [
    'heights',
    '$\\tau$ (Gyr)',
    'bins',
    'trend',
    '$/\\sigma$',
    '$\\chi^2_\\nu$',
    'trend',
    '$/\\sigma$'
  ],
  rows: [
    ...data.surveys.flatMap((survey) =>
      survey.ranges.map((r) => [
        `${NAMES[survey.name] ?? survey.name} counts`,
        `${r.lo}-${r.hi}`,
        String(r.nBins),
        `$${signed(r.tracer.value, 3)} \\pm ${fixed(r.tracer.err, 3)}$`,
        `$${signed(significance(r.tracer), 1)}$`,
        fixed(r.tracer.chi2Dof, 1),
        `$${signed(r.sheet.value, 3)} \\pm ${fixed(r.sheet.err, 3)}$`,
        `$${signed(significance(r.sheet), 1)}$`
      ])
    ),
    [
      'Imig et al.',
      `${fixed(data.imig.bins[0]?.ageLo ?? Number.NaN)}-${fixed(data.imig.bins.at(-1)?.ageHi ?? Number.NaN)}`,
      String(data.imig.bins.length),
      `$${signed(data.imig.trend.value, 3)} \\pm ${fixed(data.imig.trend.err, 3)}$`,
      `$${signed(significance(data.imig.trend), 1)}$`,
      fixed(data.imig.trend.chi2Dof, 1),
      '',
      ''
    ]
  ]
}))

/** Every age bin of Imig et al.: their pooled h_Z, the APOGEE sigma_z, and the tracer over h_Z. */
export const ImigTable = fromRecord(MILKY_WAY_PATH, parseMilkyWay, (data: MilkyWayData) => ({
  head: [
    '$\\tau$ (Gyr)',
    'populations',
    '$h_Z$ (kpc)',
    'stars',
    '$\\sigma_z$ (km s$^{-1}$)',
    'tracer'
  ],
  rows: data.imig.bins.map((b) => [
    `${fixed(b.ageLo)}-${fixed(b.ageHi)}`,
    String(b.nPopulations),
    estimateCell(b.measured.value, b.measured.err, 3),
    String(b.nKin),
    estimateCell(b.sigmaZ.value, b.sigmaZ.err, 1),
    estimateCell(b.ratio.value, b.ratio.err, 2)
  ])
}))

/** The proxy J_R,w and the whole-disk heights of every halo. */
export const JrTable = fromRecord(JR_RELATION_PATH, parseJrRelation, (data: JrRelationData) => ({
  head: [
    'halo',
    '$J_{R,w}$ (kpc km s$^{-1}$)',
    '$h_Z$, Palicio (kpc)',
    '$h_Z$, self-consistent (kpc)'
  ],
  rows: data.points.map((p) => [p.halo, fixed(p.jrW, 1), fixed(p.hzPalicio), fixed(p.hzScaled)])
}))
