import MathLabel from '@site/src/components/MathLabel'
import { between, count, fixed, percent, plusMinus } from '@site/src/components/status/format'
import {
  JR_RELATION_PATH,
  type JrRelationData,
  type Line,
  parseJrRelation
} from '@site/src/components/status/jrRelation'
import { extent, trendText } from '@site/src/components/status/metrics'
import {
  boxText,
  cellMedian,
  type ImigBin,
  type ImigCells,
  type ImigHeight,
  imigAge,
  isMeasured,
  MILKY_WAY_PATH,
  type MilkyWayData,
  type MwRange,
  type MwSurvey,
  parseMilkyWay,
  type TargetedCells
} from '@site/src/components/status/milkyWay'
import useRecord from '@site/src/components/status/useRecord'
import React, { type ReactNode } from 'react'

function lineText(line: Line): string {
  return `${plusMinus(line.slope, line.slopeErr, 0, false)} km/s`
}

/**
 * A component that quotes the numbers *metrics* computes from the record at *path*, by
 * name, so that the prose of a page cannot drift from its record.
 */
function recordMetric<T, M extends Record<string, (data: T) => string>>(
  path: string,
  parse: (record: unknown) => T,
  metrics: M
): (props: { readonly name: keyof M & string }) => ReactNode {
  return function RecordMetricValue({ name }: { readonly name: keyof M & string }): ReactNode {
    const { value, error } = useRecord(path, parse)
    if (error !== null) {
      return <span className="metric metric--error">(no data: {error})</span>
    }
    return (
      <span className="metric">
        {value === null ? '(loading)' : <MathLabel label={metrics[name](value)} />}
      </span>
    )
  }
}

function survey(data: MilkyWayData, name: string): MwSurvey {
  const found = data.surveys.find((s) => s.name === name)
  if (found === undefined) {
    throw new Error(`no survey ${name}`)
  }
  return found
}

function range(data: MilkyWayData, name: string, lo: number): MwRange {
  const found = survey(data, name).ranges.find((r) => r.lo === lo)
  if (found === undefined) {
    throw new Error(`no range from ${lo} Gyr in ${name}`)
  }
  return found
}

function ratioRange(data: MilkyWayData, name: string, which: 'ratioTracer' | 'ratioSheet'): string {
  return between(extent(survey(data, name).bins.map((b) => b[which].value)))
}

function within10(data: MilkyWayData, name: string): string {
  const ratios = survey(data, name).bins.map((b) => b.ratioTracer.value)
  return `${ratios.filter((r) => Math.abs(r - 1) <= 0.1).length} of ${ratios.length}`
}

function countHeight(data: MilkyWayData, name: string, ageLo: number): string {
  const bin = survey(data, name).bins.find((b) => b.ageLo === ageLo)
  return bin === undefined
    ? 'none'
    : `${plusMinus(bin.measured.value, bin.measured.err, 3, false)} kpc`
}

/** The Imig et al. bins whose mean age lies in [lo, hi) Gyr. */
function imigBins(data: MilkyWayData, lo = 0, hi = Number.POSITIVE_INFINITY): readonly ImigBin[] {
  return data.imig.bins.filter((b) => imigAge(b) >= lo && imigAge(b) < hi)
}

/** The range of the height `top` over the height `bottom` in the Imig et al. bins of mean age in [lo, hi) Gyr. */
function heightRatios(
  data: MilkyWayData,
  top: ImigHeight,
  bottom: ImigHeight,
  lo?: number,
  hi?: number
): string {
  return between(
    extent(imigBins(data, lo, hi).map((b) => b.heights[top].value / b.heights[bottom].value))
  )
}

/** The range of the tracer over the height `height` in the Imig et al. bins. */
function imigRatios(data: MilkyWayData, height: ImigHeight): string {
  return between(extent(data.imig.bins.map((b) => b.ratios[height].value)))
}

/** The Imig et al. bin that holds `age` Gyr, if any. */
function imigBinAt(data: MilkyWayData, age: number): ImigBin | undefined {
  return data.imig.bins.find((b) => b.ageLo <= age && age < b.ageHi)
}

/** The height `height` of the Imig et al. bin that holds `age` Gyr. */
function imigHeight(data: MilkyWayData, height: ImigHeight, age: number): string {
  const bin = imigBinAt(data, age)
  return bin === undefined
    ? 'none'
    : `${plusMinus(bin.heights[height].value, bin.heights[height].err, 3, false)} kpc`
}

/** The median of our counts over their h_Z in the measured `cells` whose [M/H] center lies in [lo, hi), and their number. */
function medianText(cells: ImigCells, lo: number, hi: number): string {
  const { median, n } = cellMedian(cells, lo, hi)
  return `${fixed(median)} over ${n} cells`
}

/** The targeting box `index` of the record and its cells. */
function targeted(data: MilkyWayData, index: number): TargetedCells {
  const found = data.imigCellsTargeting[index]
  if (found === undefined) {
    throw new Error(`no targeting box ${index}`)
  }
  return found
}

/** The ages of the Imig et al. bins without the height `height`. */
function imigMissing(data: MilkyWayData, height: ImigHeight): string {
  const missing = data.imig.bins.filter((b) => !Number.isFinite(b.heights[height].value))
  return missing.length === 0
    ? 'none'
    : missing.map((b) => `${fixed(b.ageLo)}-${fixed(b.ageHi)} Gyr`).join(', ')
}

const MW_METRICS = {
  r0: (d: MilkyWayData) => fixed(d.r0, 3),
  sigmaSurf: (d: MilkyWayData) => fixed(d.sigmaSurf, 1),
  nBootstrap: (d: MilkyWayData) => String(d.nBootstrap),
  lamostGaia: (d: MilkyWayData) => count(survey(d, 'lamost').nGaia),
  lamostMatched: (d: MilkyWayData) => count(survey(d, 'lamost').nMatched),
  apogeeGaia: (d: MilkyWayData) => count(survey(d, 'apogee').nGaia),
  apogeeMatched: (d: MilkyWayData) => count(survey(d, 'apogee').nMatched),
  distmassMatched: (d: MilkyWayData) => count(survey(d, 'apogee_distmass').nMatched),
  lamostTracer: (d: MilkyWayData) => ratioRange(d, 'lamost', 'ratioTracer'),
  apogeeTracer: (d: MilkyWayData) => ratioRange(d, 'apogee', 'ratioTracer'),
  lamostSheet: (d: MilkyWayData) => ratioRange(d, 'lamost', 'ratioSheet'),
  apogeeSheet: (d: MilkyWayData) => ratioRange(d, 'apogee', 'ratioSheet'),
  distmassTracer: (d: MilkyWayData) => ratioRange(d, 'apogee_distmass', 'ratioTracer'),
  distmassSheet: (d: MilkyWayData) => ratioRange(d, 'apogee_distmass', 'ratioSheet'),
  lamostTracerTrend0: (d: MilkyWayData) => trendText(range(d, 'lamost', 0).tracer),
  apogeeTracerTrend0: (d: MilkyWayData) => trendText(range(d, 'apogee', 0).tracer),
  lamostTracerTrend2: (d: MilkyWayData) => trendText(range(d, 'lamost', 2).tracer),
  apogeeTracerTrend2: (d: MilkyWayData) => trendText(range(d, 'apogee', 2).tracer),
  distmassTracerTrend0: (d: MilkyWayData) => trendText(range(d, 'apogee_distmass', 0).tracer),
  distmassTracerTrend2: (d: MilkyWayData) => trendText(range(d, 'apogee_distmass', 2).tracer),
  lamostSheetTrend0: (d: MilkyWayData) => trendText(range(d, 'lamost', 0).sheet),
  apogeeSheetTrend0: (d: MilkyWayData) => trendText(range(d, 'apogee', 0).sheet),
  distmassSheetTrend0: (d: MilkyWayData) => trendText(range(d, 'apogee_distmass', 0).sheet),
  lamostEta: (d: MilkyWayData) => fixed(range(d, 'lamost', 0).eta),
  apogeeEta: (d: MilkyWayData) => fixed(range(d, 'apogee', 0).eta),
  distmassEta: (d: MilkyWayData) => fixed(range(d, 'apogee_distmass', 0).eta),
  lamostWithin10: (d: MilkyWayData) => within10(d, 'lamost'),
  apogeeWithin10: (d: MilkyWayData) => within10(d, 'apogee'),
  distmassWithin10: (d: MilkyWayData) => within10(d, 'apogee_distmass'),
  lamostSheetTrend2: (d: MilkyWayData) => trendText(range(d, 'lamost', 2).sheet),
  apogeeSheetTrend2: (d: MilkyWayData) => trendText(range(d, 'apogee', 2).sheet),
  distmassSheetTrend2: (d: MilkyWayData) => trendText(range(d, 'apogee_distmass', 2).sheet),
  lamostHz68: (d: MilkyWayData) => countHeight(d, 'lamost', 6),
  apogeeHz68: (d: MilkyWayData) => countHeight(d, 'apogee', 6),
  apogeeHz24: (d: MilkyWayData) => countHeight(d, 'apogee', 2),
  distmassFailed810: (d: MilkyWayData) => {
    const bin = survey(d, 'apogee_distmass').bins.find((b) => b.ageLo === 8)
    return bin === undefined ? 'none' : percent(bin.failed, 0)
  },
  distmassHz24: (d: MilkyWayData) => countHeight(d, 'apogee_distmass', 2),
  imigAges7: (d: MilkyWayData) => {
    const bin = imigBinAt(d, 7)
    return bin === undefined ? 'none' : `${fixed(bin.ageLo)}-${fixed(bin.ageHi)} Gyr`
  },
  imigHz7: (d: MilkyWayData) => imigHeight(d, 'imigMass', 7),
  imigCountsHz7: (d: MilkyWayData) => imigHeight(d, 'imigCounts', 7),
  countsDistmassHz7: (d: MilkyWayData) => imigHeight(d, 'countsDistmass', 7),
  imigWeighting: (d: MilkyWayData) => heightRatios(d, 'imigCounts', 'imigMass'),
  imigAgeScale: (d: MilkyWayData) => heightRatios(d, 'countsAstronn', 'countsDistmass'),
  imigGap: (d: MilkyWayData) => heightRatios(d, 'countsDistmass', 'imigCounts'),
  imigGapYoung: (d: MilkyWayData) => heightRatios(d, 'countsDistmass', 'imigCounts', 0, 4),
  imigGapOld: (d: MilkyWayData) => heightRatios(d, 'countsDistmass', 'imigCounts', 4),
  imigMassRatio: (d: MilkyWayData) => imigRatios(d, 'imigMass'),
  imigCountsRatio: (d: MilkyWayData) => imigRatios(d, 'imigCounts'),
  countsAstronnRatio: (d: MilkyWayData) => imigRatios(d, 'countsAstronn'),
  countsDistmassRatio: (d: MilkyWayData) => imigRatios(d, 'countsDistmass'),
  imigMassTrend: (d: MilkyWayData) => trendText(d.imig.trends.imigMass),
  imigCountsTrend: (d: MilkyWayData) => trendText(d.imig.trends.imigCounts),
  countsAstronnTrend: (d: MilkyWayData) => trendText(d.imig.trends.countsAstronn),
  countsDistmassTrend: (d: MilkyWayData) => trendText(d.imig.trends.countsDistmass),
  countsDistmassMissing: (d: MilkyWayData) => imigMissing(d, 'countsDistmass'),
  cellsMin: (d: MilkyWayData) => String(d.imigCells.minMatches),
  cellsMeasured: (d: MilkyWayData) => `${d.imigCells.nMeasured} of ${d.imigCells.cells.length}`,
  cellsMedian: (d: MilkyWayData) => fixed(d.imigCells.medianRatio),
  cellsTrend: (d: MilkyWayData) => trendText(d.imigCells.trend),
  cellsMetalPoor: (d: MilkyWayData) => medianText(d.imigCells, Number.NEGATIVE_INFINITY, -0.1),
  cellsMetalSolar: (d: MilkyWayData) => medianText(d.imigCells, -0.1, 0.1),
  cellsMetalRich: (d: MilkyWayData) => medianText(d.imigCells, 0.1, Number.POSITIVE_INFINITY),
  cellsMatches: (d: MilkyWayData) => count(d.imigCells.nMatches),
  boxA: (d: MilkyWayData) => boxText(targeted(d, 0).box),
  boxB: (d: MilkyWayData) => boxText(targeted(d, 1).box),
  boxMatchesA: (d: MilkyWayData) => count(targeted(d, 0).cells.nMatches),
  boxMatchesB: (d: MilkyWayData) => count(targeted(d, 1).cells.nMatches),
  boxPoorA: (d: MilkyWayData) => medianText(targeted(d, 0).cells, Number.NEGATIVE_INFINITY, -0.1),
  boxPoorB: (d: MilkyWayData) => medianText(targeted(d, 1).cells, Number.NEGATIVE_INFINITY, -0.1),
  boxRichA: (d: MilkyWayData) => medianText(targeted(d, 0).cells, 0.1, Number.POSITIVE_INFINITY),
  boxRichB: (d: MilkyWayData) => medianText(targeted(d, 1).cells, 0.1, Number.POSITIVE_INFINITY),
  boxTrendA: (d: MilkyWayData) => trendText(targeted(d, 0).cells.trend),
  boxTrendB: (d: MilkyWayData) => trendText(targeted(d, 1).cells.trend),
  slopeCountsApogee: (d: MilkyWayData) => trendText(d.imigCells.slopes.counts),
  slopeImigApogee: (d: MilkyWayData) => trendText(d.imigCells.slopes.imig),
  slopeCountsLamost: (d: MilkyWayData) => trendText(d.imigCellsLamost.slopes.counts),
  slopeImigLamost: (d: MilkyWayData) => trendText(d.imigCellsLamost.slopes.imig),
  lamostCells: (d: MilkyWayData) => String(d.imigCellsLamost.nMeasured),
  lamostCellMatches: (d: MilkyWayData) => count(d.imigCellsLamost.nMatches),
  cellsStars: (d: MilkyWayData) => {
    const measured = d.imigCells.cells.filter((c) => isMeasured(d.imigCells, c))
    const sum = (cells: readonly { readonly nStars: number }[]) =>
      cells.reduce((total, c) => total + c.nStars, 0)
    return `${count(sum(measured))} of their ${count(sum(d.imigCells.cells))} stars`
  }
} satisfies Record<string, (d: MilkyWayData) => string>

const JR_METRICS = {
  slopePalicio: (d: JrRelationData) => lineText(d.linePalicio),
  slopeScaled: (d: JrRelationData) => lineText(d.lineScaled),
  palicioSlope: (d: JrRelationData) => fixed(d.palicioSlope, 1),
  jrRange: (d: JrRelationData) => between(extent(d.points.map((p) => p.jrW)), 1),
  hzPalicioRange: (d: JrRelationData) => between(extent(d.points.map((p) => p.hzPalicio))),
  hzScaledRange: (d: JrRelationData) => between(extent(d.points.map((p) => p.hzScaled))),
  slopeRatio: (d: JrRelationData) => fixed(d.linePalicio.slope / d.palicioSlope),
  largestShifts: (d: JrRelationData) =>
    [...d.points]
      .sort((a, b) => b.hzPalicio / b.hzScaled - a.hzPalicio / a.hzScaled)
      .slice(0, 2)
      .map((p) => `${p.halo} (${fixed(p.hzPalicio / p.hzScaled)})`)
      .join(' and '),
  scaledDeviation: (d: JrRelationData) =>
    fixed(Math.abs(d.lineScaled.slope - d.palicioSlope) / d.lineScaled.slopeErr, 1),
  palicioDeviation: (d: JrRelationData) =>
    fixed(Math.abs(d.linePalicio.slope - d.palicioSlope) / d.linePalicio.slopeErr, 1),
  slopePalicioMixture: (d: JrRelationData) => lineText(d.linePalicioMixture),
  slopeScaledMixture: (d: JrRelationData) => lineText(d.lineScaledMixture),
  scaledMixtureShift: (d: JrRelationData) =>
    fixed(Math.abs(d.lineScaledMixture.slope - d.lineScaled.slope) / d.lineScaled.slopeErr, 1),
  scaledMixtureDeviation: (d: JrRelationData) =>
    fixed(Math.abs(d.lineScaledMixture.slope - d.palicioSlope) / d.lineScaledMixture.slopeErr, 1)
} satisfies Record<string, (d: JrRelationData) => string>

/** A number of the Milky Way page, computed from its record. */
export const MwMetric = recordMetric(MILKY_WAY_PATH, parseMilkyWay, MW_METRICS)

/** A number of the page across galaxies, computed from its record. */
export const JrMetric = recordMetric(JR_RELATION_PATH, parseJrRelation, JR_METRICS)
