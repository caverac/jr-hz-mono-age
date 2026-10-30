import { Pending } from '@site/src/components/status/controls'
import {
  JR_RELATION_PATH,
  type JrRelationData,
  parseJrRelation
} from '@site/src/components/status/jrRelation'
import { median, significance } from '@site/src/components/status/metrics'
import {
  boxText,
  cellMedian,
  IMIG_HEIGHTS,
  type ImigCells,
  type ImigHeight,
  isMeasured,
  measuredIn,
  METALLICITY_LABELS,
  METALLICITY_RANGES,
  MILKY_WAY_PATH,
  type MilkyWayData,
  parseMilkyWay,
  surveyName
} from '@site/src/components/status/milkyWay'
import type { Trend } from '@site/src/components/status/record'
import { fixed, signed, Table, type TableProps } from '@site/src/components/status/tables'
import useRecord from '@site/src/components/status/useRecord'
import React, { type ReactNode } from 'react'

/** A slope with its error, or "none" where the record has none. */
function trendCell(t: Trend): string {
  return Number.isFinite(t.value) ? `$${signed(t.value)} \\pm ${fixed(t.err)}$` : 'none'
}

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
    '$g$ (kpc$^{-1}$)',
    '$h_Z$ (kpc)',
    'tracer',
    'sheet',
    'Jeans'
  ],
  rows: data.surveys.flatMap((survey) =>
    survey.bins.map((b) => [
      surveyName(survey.name),
      `${b.ageLo}-${b.ageHi}`,
      fixed(b.ageMedian),
      String(b.nMatched),
      String(b.nKin),
      estimateCell(b.sigmaZ.value, b.sigmaZ.err, 1),
      Number.isFinite(b.gradient) ? `$${signed(b.gradient)}$` : 'none',
      estimateCell(b.measured.value, b.measured.err, 3),
      estimateCell(b.ratioTracer.value, b.ratioTracer.err, 2),
      estimateCell(b.ratioSheet.value, b.ratioSheet.err, 2),
      estimateCell(b.ratioJeans.value, b.ratioJeans.err, 2)
    ])
  )
}))

/**
 * The potentials around McMillan (2017) within one error of the measured rho_0 and
 * Sigma_1.1, after McMillan's own: their changes, densities and v_c at R_0, the median of
 * the tracer over h_Z on each age scale, and the growth of the exponential fit to the tracer.
 */
export const FamilyTable = fromRecord(MILKY_WAY_PATH, parseMilkyWay, (data: MilkyWayData) => {
  const members = data.family
    .map((member, index) => ({ member, index }))
    .filter(
      ({ member: m }) =>
        m.admissible || (m.thinFactor === 1 && m.thinHeight === 0.3 && m.haloFactor === 1)
    )
    .sort((a, b) => Number(a.member.admissible) - Number(b.member.admissible))
  return {
    groups: [
      ['', 3],
      ['at $R_0$', 3],
      ['tracer / $h_Z$', 3],
      ['growth', 2]
    ],
    head: [
      'thin disk',
      '$h$ (kpc)',
      'halo',
      '$\\rho_0$ ($M_\\odot$ pc$^{-3}$)',
      '$\\Sigma_{1.1}$ ($M_\\odot$ pc$^{-2}$)',
      '$v_c$ (km s$^{-1}$)',
      ...data.surveys.map((s) => surveyName(s.name)),
      '$|z| < 1.2$ kpc',
      'whole'
    ],
    rows: members.map(({ member: m, index }) => [
      `$\\times ${fixed(m.thinFactor, 1)}$`,
      fixed(m.thinHeight),
      `$\\times ${fixed(m.haloFactor, 1)}$`,
      `$${fixed(m.rho0, 3)}$`,
      `$${fixed(m.sigma11, 1)}$`,
      `$${fixed(m.vCirc, 0)}$`,
      ...data.surveys.map((s) => `$${fixed(median(s.bins.map((b) => b.ratioFamily[index])))}$`),
      `$${fixed(m.growthWindow)}$`,
      `$${fixed(m.growthWhole)}$`
    ])
  }
})

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
      const ratios = measured.map((c) => c.ratios.counted.value)
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

/**
 * The cells with all the matches and with those of each targeting box: the matches, the
 * cells measured, the median of our counts over their h_Z in three ranges of [M/H], and the
 * slope of its logarithm.
 */
export const TargetingTable = fromRecord(MILKY_WAY_PATH, parseMilkyWay, (data: MilkyWayData) => {
  const row = (selection: string, cells: ImigCells): readonly string[] => [
    selection,
    String(cells.nMatches),
    String(cells.nMeasured),
    ...METALLICITY_RANGES.map(([lo, hi]) => fixed(cellMedian(cells, lo, hi).median)),
    trendCell(cells.trends.counted.metallicity)
  ]
  return {
    groups: [
      ['', 3],
      ['median counts over Imig et al.', 3],
      ['', 1]
    ],
    head: ['matches used', 'matches', 'cells', ...METALLICITY_LABELS, 'per dex, fixed age'],
    rows: [
      row('all', data.imigCells),
      ...data.imigCellsTargeting.map((t) => row(boxText(t.box), t.cells))
    ]
  }
})

/**
 * The slope of ln h_Z per dex of [M/H] at fixed age, in the measured cells of each set: our
 * windowed counts, the exponential fit to them, the published heights of Imig et al., the
 * tracer of each cell's sigma_z, and ln sigma_z itself.
 */
export const SlopeTable = fromRecord(MILKY_WAY_PATH, parseMilkyWay, (data: MilkyWayData) => {
  const row = (counts: string, cells: ImigCells): readonly string[] => [
    counts,
    String(cells.nMeasured),
    trendCell(cells.slopes.counts),
    trendCell(cells.slopes.countsExponential),
    trendCell(cells.slopes.imigSun),
    trendCell(cells.slopes.tracer),
    trendCell(cells.slopes.sigmaZ)
  ]
  return {
    groups: [
      ['', 2],
      ['our counts', 2],
      ['', 2],
      ['', 1]
    ],
    head: [
      'counts',
      'cells',
      'windowed',
      'exponential fit',
      'Imig et al.',
      'tracer',
      '$\\sigma_z$'
    ],
    rows: [
      row('APOGEE, all matches', data.imigCells),
      ...data.imigCellsTargeting.map((t) => row(`APOGEE, ${boxText(t.box)}`, t.cells)),
      row('LAMOST', data.imigCellsLamost)
    ]
  }
})

/**
 * The windowed h_Z of the tracer of each sigma_z against the exponential fits to the same
 * profile below the edge of the counts and over the whole profile.
 */
export const EstimatorTable = fromRecord(MILKY_WAY_PATH, parseMilkyWay, (data: MilkyWayData) => ({
  groups: [
    ['', 2],
    ['exponential fit (kpc)', 2],
    ['ratios', 2]
  ],
  head: [
    '$\\sigma_z$ (km s$^{-1}$)',
    'windowed $h_Z$ (kpc)',
    '$|z| < 1.2$ kpc',
    'whole profile',
    'windowed over fit',
    'whole over $|z| < 1.2$ kpc'
  ],
  rows: data.estimators.map((e) => [
    fixed(e.sigmaZ, 0),
    fixed(e.windowed, 3),
    fixed(e.exponential, 3),
    fixed(e.exponentialWhole, 3),
    fixed(e.windowed / e.exponential),
    fixed(e.exponentialWhole / e.exponential)
  ])
}))

/**
 * How the three ratios of the cells change, in each set of cells: the slope of their
 * logarithm per ln age at fixed [M/H], and per dex of [M/H] at fixed age.
 */
export const RatioTrendTable = fromRecord(MILKY_WAY_PATH, parseMilkyWay, (data: MilkyWayData) => {
  const row = (counts: string, cells: ImigCells): readonly string[] => [
    counts,
    String(cells.nMeasured),
    ...(['counted', 'exponential', 'tracer'] as const).flatMap((ratio) => [
      trendCell(cells.trends[ratio].age),
      trendCell(cells.trends[ratio].metallicity)
    ])
  ]
  return {
    groups: [
      ['', 2],
      ['windowed, ours over theirs', 2],
      ['exponential fit, ours over theirs', 2],
      ['tracer over ours', 2]
    ],
    head: [
      'counts',
      'cells',
      'per ln age',
      'per dex',
      'per ln age',
      'per dex',
      'per ln age',
      'per dex'
    ],
    rows: [
      row('APOGEE, all matches', data.imigCells),
      ...data.imigCellsTargeting.map((t) => row(`APOGEE, ${boxText(t.box)}`, t.cells)),
      row('LAMOST', data.imigCellsLamost)
    ]
  }
})

/**
 * The measured APOGEE cells of each range of [M/H], as medians: our windowed h_Z over their
 * model through our counts, the exponential fit to our counts over that to their model
 * through our counts, and the tracer of each cell's sigma_z over our windowed h_Z.
 */
export const ImigEstimatorTable = fromRecord(
  MILKY_WAY_PATH,
  parseMilkyWay,
  (data: MilkyWayData) => ({
    groups: [
      ['', 2],
      ['ours over Imig et al.', 2],
      ['', 1]
    ],
    head: ['$[\\mathrm{M/H}]$', 'cells', 'windowed', 'exponential fit', 'tracer over ours'],
    rows: METALLICITY_RANGES.map(([lo, hi], j) => {
      const cells = measuredIn(data.imigCells, lo, hi)
      return [
        METALLICITY_LABELS[j] ?? '',
        String(cells.length),
        fixed(cellMedian(data.imigCells, lo, hi, 'counted').median),
        fixed(cellMedian(data.imigCells, lo, hi, 'exponential').median),
        fixed(cellMedian(data.imigCells, lo, hi, 'tracer').median)
      ]
    })
  })
)

/** A ratio with three digits, or "none" where the record has none. */
function ratioCell(value: number): string {
  return Number.isFinite(value) ? fixed(value, 3) : 'none'
}

/**
 * What the counts give an exponential of each scale, over the scale, for the Gaia stars of
 * each survey: the red clump of LAMOST and the giant box of APOGEE, whose two age scales
 * share it.
 */
export const CountBiasTable = fromRecord(MILKY_WAY_PATH, parseMilkyWay, (data: MilkyWayData) => {
  const samples = [
    { name: 'LAMOST red clump', survey: 'lamost' },
    { name: 'APOGEE giant box', survey: 'apogee' }
  ]
  const scales = data.surveys[0]?.bias.scales ?? []
  return {
    groups: [
      ['', 1],
      ['scale of the exponential (kpc)', scales.length]
    ],
    head: ['Gaia stars', ...scales.map((scale) => fixed(scale))],
    rows: samples.map(({ name, survey }) => [
      name,
      ...(data.surveys.find((s) => s.name === survey)?.bias.ratios ?? []).map(ratioCell)
    ])
  }
})

/**
 * Their heights in the measured APOGEE cells of each range of [M/H], as medians: their model
 * over our cylinder in the continuous window over their h_Z at their Sun, the same model
 * through our counts over the continuous window, and our counts over their model through
 * our counts.
 */
export const ImigModelTable = fromRecord(MILKY_WAY_PATH, parseMilkyWay, (data: MilkyWayData) => ({
  groups: [
    ['', 2],
    ['their model', 2],
    ['', 1]
  ],
  head: [
    '$[\\mathrm{M/H}]$',
    'cells',
    'cylinder over published',
    'counts over cylinder',
    'ours over theirs'
  ],
  rows: METALLICITY_RANGES.map(([lo, hi], j) => {
    const cells = measuredIn(data.imigCells, lo, hi)
    return [
      METALLICITY_LABELS[j] ?? '',
      String(cells.length),
      ratioCell(median(cells.map((c) => c.theirs.window / c.theirs.sun.value))),
      ratioCell(median(cells.map((c) => c.theirs.counted.value / c.theirs.window))),
      fixed(median(cells.map((c) => c.ratios.counted.value)))
    ]
  })
}))

/**
 * Our counts over their model through our counts, in each slice of |z|, each normalized to
 * its sum: the median over the measured APOGEE cells of each range of [M/H].
 */
export const ImigShapeTable = fromRecord(MILKY_WAY_PATH, parseMilkyWay, (data: MilkyWayData) => {
  const edges = data.sliceEdges
  const slices = edges
    .slice(1)
    .map((hi, i) => `${fixed(edges[i] ?? Number.NaN, 1)}-${fixed(hi, 1)}`)
  return {
    groups: [
      ['', 1],
      ['$|z|$ (kpc)', slices.length]
    ],
    head: ['$[\\mathrm{M/H}]$', ...slices],
    rows: METALLICITY_RANGES.map(([lo, hi], j) => {
      const shapes = measuredIn(data.imigCells, lo, hi).map((c) => c.ours.shape)
      return [
        METALLICITY_LABELS[j] ?? '',
        ...slices.map((_, i) => fixed(median(shapes.map((shape) => shape[i] ?? Number.NaN))))
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
