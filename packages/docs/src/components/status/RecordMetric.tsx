import type { Trend } from '@site/src/components/status/data'
import {
  JR_RELATION_PATH,
  type JrRelationData,
  type Line,
  parseJrRelation
} from '@site/src/components/status/jrRelation'
import { extent, median, significance } from '@site/src/components/status/metrics'
import {
  MILKY_WAY_PATH,
  type MilkyWayData,
  type MwRange,
  type MwSurvey,
  parseMilkyWay
} from '@site/src/components/status/milkyWay'
import useRecord from '@site/src/components/status/useRecord'
import React, { type ReactNode } from 'react'

function fixed(value: number, digits = 2): string {
  return value.toFixed(digits)
}

function signed(value: number, digits = 3): string {
  return `${value >= 0 ? '+' : '-'}${Math.abs(value).toFixed(digits)}`
}

function between([lo, hi]: readonly [number, number], digits = 2): string {
  return `${fixed(lo, digits)} to ${fixed(hi, digits)}`
}

function trendText(trend: Trend): string {
  return `${signed(trend.value)} +/- ${fixed(trend.err, 3)} (${signed(significance(trend), 1)} standard errors)`
}

function lineText(line: Line): string {
  return `${fixed(line.slope, 0)} +/- ${fixed(line.slopeErr, 0)} km/s`
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
    return <span className="metric">{value === null ? '(loading)' : metrics[name](value)}</span>
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

const MW_METRICS = {
  r0: (d: MilkyWayData) => fixed(d.r0, 3),
  sigmaSurf: (d: MilkyWayData) => fixed(d.sigmaSurf, 1),
  nBootstrap: (d: MilkyWayData) => String(d.nBootstrap),
  lamostGaia: (d: MilkyWayData) => survey(d, 'lamost').nGaia.toLocaleString('en-US'),
  lamostMatched: (d: MilkyWayData) => survey(d, 'lamost').nMatched.toLocaleString('en-US'),
  apogeeGaia: (d: MilkyWayData) => survey(d, 'apogee').nGaia.toLocaleString('en-US'),
  apogeeMatched: (d: MilkyWayData) => survey(d, 'apogee').nMatched.toLocaleString('en-US'),
  lamostTracer: (d: MilkyWayData) => ratioRange(d, 'lamost', 'ratioTracer'),
  apogeeTracer: (d: MilkyWayData) => ratioRange(d, 'apogee', 'ratioTracer'),
  lamostSheet: (d: MilkyWayData) => ratioRange(d, 'lamost', 'ratioSheet'),
  apogeeSheet: (d: MilkyWayData) => ratioRange(d, 'apogee', 'ratioSheet'),
  lamostTracerTrend0: (d: MilkyWayData) => trendText(range(d, 'lamost', 0).tracer),
  apogeeTracerTrend0: (d: MilkyWayData) => trendText(range(d, 'apogee', 0).tracer),
  lamostTracerTrend2: (d: MilkyWayData) => trendText(range(d, 'lamost', 2).tracer),
  apogeeTracerTrend2: (d: MilkyWayData) => trendText(range(d, 'apogee', 2).tracer),
  lamostSheetTrend0: (d: MilkyWayData) => trendText(range(d, 'lamost', 0).sheet),
  apogeeSheetTrend0: (d: MilkyWayData) => trendText(range(d, 'apogee', 0).sheet),
  lamostEta: (d: MilkyWayData) => fixed(range(d, 'lamost', 0).eta),
  apogeeEta: (d: MilkyWayData) => fixed(range(d, 'apogee', 0).eta),
  imigRatio: (d: MilkyWayData) => between(extent(d.imig.bins.map((b) => b.ratio.value))),
  imigRatioMedian: (d: MilkyWayData) => fixed(median(d.imig.bins.map((b) => b.ratio.value))),
  imigTrend: (d: MilkyWayData) => trendText(d.imig.trend),
  imigBins: (d: MilkyWayData) => String(d.imig.bins.length)
} satisfies Record<string, (d: MilkyWayData) => string>

const JR_METRICS = {
  slopePalicio: (d: JrRelationData) => lineText(d.linePalicio),
  slopeScaled: (d: JrRelationData) => lineText(d.lineScaled),
  palicioSlope: (d: JrRelationData) => fixed(d.palicioSlope, 1),
  jrRange: (d: JrRelationData) => between(extent(d.points.map((p) => p.jrW)), 1),
  hzPalicioRange: (d: JrRelationData) => between(extent(d.points.map((p) => p.hzPalicio))),
  hzScaledRange: (d: JrRelationData) => between(extent(d.points.map((p) => p.hzScaled))),
  slopeRatio: (d: JrRelationData) => fixed(d.linePalicio.slope / d.palicioSlope),
  palicioDeviation: (d: JrRelationData) =>
    fixed(Math.abs(d.linePalicio.slope - d.palicioSlope) / d.linePalicio.slopeErr, 1)
} satisfies Record<string, (d: JrRelationData) => string>

/** A number of the Milky Way page, computed from its record. */
export const MwMetric = recordMetric(MILKY_WAY_PATH, parseMilkyWay, MW_METRICS)

/** A number of the page across galaxies, computed from its record. */
export const JrMetric = recordMetric(JR_RELATION_PATH, parseJrRelation, JR_METRICS)
