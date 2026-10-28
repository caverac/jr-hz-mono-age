import { Pending } from '@site/src/components/status/controls'
import {
  JR_RELATION_PATH,
  type JrRelationData,
  parseJrRelation
} from '@site/src/components/status/jrRelation'
import { median, significance } from '@site/src/components/status/metrics'
import {
  IMIG_HEIGHTS,
  type ImigHeight,
  isMeasured,
  MILKY_WAY_PATH,
  type MilkyWayData,
  parseMilkyWay,
  surveyName
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

function estimateCell(value: number, err: number, digits: number): string {
  return Number.isFinite(value) ? `$${fixed(value, digits)} \\pm ${fixed(err, digits)}$` : 'none'
}

/** The bins of Imig et al. that enter the trend of the tracer over `height`: those with a finite ratio and error. */
function imigTrendBins(data: MilkyWayData, height: ImigHeight) {
  return data.imig.bins.filter((b) => {
    const ratio = b.ratios[height]
    return Number.isFinite(ratio.value) && Number.isFinite(ratio.err) && ratio.err > 0
  })
}

/** Every age bin of the three surveys of the Milky Way. */
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
      surveyName(survey.name),
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

/** The trends of h_pred / h_Z in the three surveys of the Milky Way and against the four heights of the Imig et al. bins. */
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
        `${surveyName(survey.name)} counts`,
        `${r.lo}-${r.hi}`,
        String(r.nBins),
        `$${signed(r.tracer.value, 3)} \\pm ${fixed(r.tracer.err, 3)}$`,
        `$${signed(significance(r.tracer), 1)}$`,
        fixed(r.tracer.chi2Dof, 1),
        `$${signed(r.sheet.value, 3)} \\pm ${fixed(r.sheet.err, 3)}$`,
        `$${signed(significance(r.sheet), 1)}$`
      ])
    ),
    ...IMIG_HEIGHTS.map(({ height, name }) => {
      const bins = imigTrendBins(data, height)
      const trend = data.imig.trends[height]
      return [
        `${name}, Imig bins`,
        `${fixed(bins[0]?.ageLo ?? Number.NaN)}-${fixed(bins.at(-1)?.ageHi ?? Number.NaN)}`,
        String(bins.length),
        `$${signed(trend.value, 3)} \\pm ${fixed(trend.err, 3)}$`,
        `$${signed(significance(trend), 1)}$`,
        fixed(trend.chi2Dof, 1),
        '',
        ''
      ]
    })
  ]
}))

/** The column heads of the four heights of an Imig et al. bin, under the groups of `imigGroups`. */
const IMIG_HEADS = ['by mass', 'by counts', 'astroNN ages', 'distmass ages']

/** The groups of the four heights of an Imig et al. bin, after `leading` columns. */
function imigGroups(leading: number, prefix: string): readonly (readonly [string, number])[] {
  return [
    ['', leading],
    [`${prefix}Imig et al.`, 2],
    [`${prefix}star counts`, 2]
  ]
}

/** Every age bin of Imig et al.: their populations pooled by mass and by counts, and our counts on both age scales. */
export const ImigTable = fromRecord(MILKY_WAY_PATH, parseMilkyWay, (data: MilkyWayData) => ({
  groups: imigGroups(2, '$h_Z$ (kpc), '),
  head: ['$\\tau$ (Gyr)', 'populations', ...IMIG_HEADS],
  rows: data.imig.bins.map((b) => [
    `${fixed(b.ageLo)}-${fixed(b.ageHi)}`,
    String(b.nPopulations),
    ...IMIG_HEIGHTS.map(({ height }) =>
      estimateCell(b.heights[height].value, b.heights[height].err, 3)
    )
  ])
}))

/** Every age bin of Imig et al.: the APOGEE sigma_z of its ages, the tracer, and the tracer over each height. */
export const ImigRatioTable = fromRecord(MILKY_WAY_PATH, parseMilkyWay, (data: MilkyWayData) => ({
  groups: imigGroups(4, 'tracer over '),
  head: ['$\\tau$ (Gyr)', 'stars', '$\\sigma_z$ (km s$^{-1}$)', 'tracer (kpc)', ...IMIG_HEADS],
  rows: data.imig.bins.map((b) => [
    `${fixed(b.ageLo)}-${fixed(b.ageHi)}`,
    String(b.nKin),
    estimateCell(b.sigmaZ.value, b.sigmaZ.err, 1),
    fixed(b.tracer, 3),
    ...IMIG_HEIGHTS.map(({ height }) =>
      estimateCell(b.ratios[height].value, b.ratios[height].err, 2)
    )
  ])
}))

/**
 * The populations of Imig et al. one at a time, by [M/H] bin: the cells with enough matches
 * for our h_Z, their stars and ours, and the median and range of our counts over their h_Z.
 */
export const ImigCellTable = fromRecord(MILKY_WAY_PATH, parseMilkyWay, (data: MilkyWayData) => {
  const { imigCells } = data
  const edges = [...new Set(imigCells.cells.map((c) => c.mhLo))].sort((a, b) => a - b)
  return {
    groups: [
      ['', 4],
      ['counts over Imig et al.', 2]
    ],
    head: ['$[\\mathrm{M/H}]$', 'cells', 'their stars', 'our matches', 'median', 'range'],
    rows: edges.map((lo) => {
      const all = imigCells.cells.filter((c) => c.mhLo === lo)
      const measured = all.filter((c) => isMeasured(imigCells, c))
      const ratios = measured.map((c) => c.ratio.value)
      return [
        `${signed(lo, 1)} to ${signed(all[0]?.mhHi ?? Number.NaN, 1)}`,
        `${measured.length} of ${all.length}`,
        String(measured.reduce((sum, c) => sum + c.nStars, 0)),
        String(measured.reduce((sum, c) => sum + c.nMatched, 0)),
        fixed(median(ratios)),
        measured.length < 2
          ? fixed(ratios[0] ?? Number.NaN)
          : `${fixed(Math.min(...ratios))} to ${fixed(Math.max(...ratios))}`
      ]
    })
  }
})

/** The proxy J_R,w and the whole-disk heights of every halo. */
export const JrTable = fromRecord(JR_RELATION_PATH, parseJrRelation, (data: JrRelationData) => ({
  groups: [
    ['', 2],
    ['$h_Z$, Palicio (kpc)', 2],
    ['$h_Z$, self-consistent (kpc)', 2]
  ],
  head: ['halo', '$J_{R,w}$ (kpc km s$^{-1}$)', 'measured', 'predicted', 'measured', 'predicted'],
  rows: data.points.map((p) => [
    p.halo,
    fixed(p.jrW, 1),
    fixed(p.hzPalicio),
    fixed(p.hzPalicioMixture),
    fixed(p.hzScaled),
    fixed(p.hzScaledMixture)
  ])
}))
