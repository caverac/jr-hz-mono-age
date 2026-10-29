import MathLabel from '@site/src/components/MathLabel'
import { Chips, Pending, toggled } from '@site/src/components/status/controls'
import type { Estimate } from '@site/src/components/status/data'
import {
  IMIG_HEIGHTS,
  imigAge,
  type ImigHeight,
  MILKY_WAY_PATH,
  type MilkyWayData,
  parseMilkyWay,
  surveyName
} from '@site/src/components/status/milkyWay'
import useRecord from '@site/src/components/status/useRecord'
import Chart, { Axis, linePath, ticksWithin } from '@site/src/components/viz/Chart'
import { useElementWidth } from '@site/src/components/viz/hooks'
import {
  type Dash,
  dashClass,
  Legend,
  type LegendEntry,
  Marker,
  type MarkerShape,
  seriesColor,
  Swatch
} from '@site/src/components/viz/marks'
import * as d3 from 'd3'
import React, { type ReactNode, useState } from 'react'

const HEIGHT = 340
const AGE_DOMAIN: readonly [number, number] = [0, 10]
const RATIO_TICKS = [0.3, 0.5, 0.7, 1, 1.5, 2]
const TOLERANCE = 0.1

/** One series of the chart: a set of heights and a prediction, with its points (median age, ratio). */
interface Series {
  readonly key: string
  /** The color of its line, markers and button, from the slot of the series. */
  readonly color: string
  readonly label: string
  readonly prediction: string
  readonly marker: MarkerShape
  readonly open: boolean
  readonly dash: Dash
  readonly points: readonly {
    readonly age: number
    readonly ratio: Estimate
    readonly label: string
  }[]
}

const SHAPES: Record<string, MarkerShape> = {
  lamost: 'circle',
  apogee: 'square',
  apogee_distmass: 'triangle'
}

/** The markers of the heights of Imig et al. that the chart compares with the tracer. */
const IMIG_MARKERS: Readonly<Partial<Record<ImigHeight, MarkerShape>>> = {
  imigMass: 'diamond',
  imigCounts: 'cross'
}

function seriesOf(data: MilkyWayData): readonly Series[] {
  const surveys = data.surveys.flatMap((survey) =>
    (['tracer', 'sheet'] as const).map((prediction) => ({
      key: `${survey.name}-${prediction}`,
      label: `${surveyName(survey.name)}, ${prediction}`,
      prediction,
      marker: SHAPES[survey.name] ?? 'circle',
      open: prediction === 'sheet',
      dash: (prediction === 'sheet' ? 'dashed' : 'solid') as Dash,
      points: survey.bins
        .map((bin) => ({
          age: bin.ageMedian,
          ratio: prediction === 'tracer' ? bin.ratioTracer : bin.ratioSheet,
          label: `${surveyName(survey.name)}, ${bin.ageLo}-${bin.ageHi} Gyr`
        }))
        .filter((p) => Number.isFinite(p.age) && Number.isFinite(p.ratio.value))
    }))
  )
  const imig = IMIG_HEIGHTS.filter((s) => IMIG_MARKERS[s.height] !== undefined).map(
    (s): Omit<Series, 'color'> => ({
      key: `${s.height}-tracer`,
      label: `${s.name}, tracer`,
      prediction: 'tracer',
      marker: IMIG_MARKERS[s.height] ?? 'diamond',
      open: false,
      dash: 'dotted',
      points: data.imig.bins
        .map((bin) => ({
          age: imigAge(bin),
          ratio: bin.ratios[s.height],
          label: `${s.name}, ${bin.ageLo.toFixed(2)}-${bin.ageHi.toFixed(2)} Gyr`
        }))
        .filter((p) => Number.isFinite(p.ratio.value))
    })
  )
  return [...surveys, ...imig].map((s, i) => ({ ...s, color: seriesColor(i) }))
}

interface Hovered {
  readonly color: string
  readonly label: string
  readonly prediction: string
  readonly ratio: Estimate
}

function Plot({
  series,
  width,
  onHover
}: {
  readonly series: readonly Series[]
  readonly width: number
  readonly onHover: (hovered: Hovered | null) => void
}): ReactNode {
  const values = series.flatMap((s) => s.points.map((p) => p.ratio.value)).filter(Number.isFinite)
  const domain: readonly [number, number] = [
    Math.min(0.8, d3.min(values) ?? 1) / 1.15,
    Math.max(1.25, d3.max(values) ?? 1) * 1.15
  ]
  return (
    <Chart width={width} height={HEIGHT} xLabel="$\tau$ (Gyr)" yLabel="$h_\mathrm{pred}/h_Z$">
      {(inner) => {
        const x = d3.scaleLinear().domain(AGE_DOMAIN).range([0, inner.width])
        const y = d3.scaleLog().domain(domain).range([inner.height, 0])
        return (
          <>
            <rect
              x={0}
              width={inner.width}
              y={y(1 + TOLERANCE)}
              height={y(1 - TOLERANCE) - y(1 + TOLERANCE)}
              className="viz-band"
            />
            <Axis
              scale={y}
              orientation="left"
              tickValues={ticksWithin(RATIO_TICKS, domain)}
              tickFormat={(v) => String(v)}
              gridLength={inner.width}
            />
            <Axis
              scale={x}
              orientation="bottom"
              transform={`translate(0,${inner.height})`}
              ticks={5}
              gridLength={inner.height}
            />
            <line x1={0} x2={inner.width} y1={y(1)} y2={y(1)} className="viz-line viz-line--thin" />
            {series.map((s) => (
              <g key={s.key}>
                <path
                  d={linePath(s.points.map((p) => [x(p.age), y(p.ratio.value)] as const))}
                  className={`${dashClass(s.dash)} viz-line--thin`}
                  style={{ stroke: s.color }}
                />
                {s.points.map((p) => (
                  <g key={p.label}>
                    <line
                      x1={x(p.age)}
                      x2={x(p.age)}
                      y1={y(Math.max(p.ratio.value - p.ratio.err, domain[0]))}
                      y2={y(Math.min(p.ratio.value + p.ratio.err, domain[1]))}
                      className="viz-line viz-line--thin"
                      style={{ stroke: s.color }}
                    />
                    <Marker
                      shape={s.marker}
                      open={s.open}
                      color={s.color}
                      x={x(p.age)}
                      y={y(p.ratio.value)}
                      onMouseEnter={() =>
                        onHover({
                          color: s.color,
                          label: p.label,
                          prediction: s.prediction,
                          ratio: p.ratio
                        })
                      }
                      onMouseLeave={() => onHover(null)}
                    />
                  </g>
                ))}
              </g>
            ))}
          </>
        )
      }}
    </Chart>
  )
}

/**
 * The scale height that the isothermal tracer and the sheet predict for every mono-age
 * population of the Milky Way, over the star-count h_Z of LAMOST and of APOGEE on both age
 * scales, and the tracer over the heights of Imig et al. (2025) pooled by mass and by counts,
 * against the median age.
 */
export default function MilkyWayChart(): ReactNode {
  const [ref, width] = useElementWidth<HTMLDivElement>()
  const { value, error } = useRecord(MILKY_WAY_PATH, parseMilkyWay)
  const [hidden, setHidden] = useState<ReadonlySet<string>>(new Set())
  const [hovered, setHovered] = useState<Hovered | null>(null)
  if (value === null) {
    return (
      <div className="viz" ref={ref}>
        <Pending error={error} />
      </div>
    )
  }
  const all = seriesOf(value)
  const shown = all.filter((s) => !hidden.has(s.key))
  const entries: readonly LegendEntry[] = [
    ...shown.map((s) => ({
      label: s.label,
      marker: s.marker,
      open: s.open,
      dash: s.dash,
      color: s.color
    })),
    { label: `$|h_\\mathrm{pred}/h_Z - 1| \\le ${TOLERANCE}$`, band: true }
  ]
  return (
    <div className="viz" ref={ref}>
      <Plot series={shown} width={width} onHover={setHovered} />
      <Legend entries={entries} />
      <div className="viz-controls">
        <Chips
          label="Series"
          options={all.map((s) => [s.key, s.label] as const)}
          isPressed={(key) => !hidden.has(key)}
          onToggle={(key) => setHidden((current) => toggled(current, key))}
          swatch={(key) => all.find((s) => s.key === key)?.color}
        />
      </div>
      <p className="viz-note">
        {hovered === null ? (
          'Point at a population to read its ratio.'
        ) : (
          <>
            <Swatch color={hovered.color} />
            <MathLabel
              label={`${hovered.label}, ${hovered.prediction}: $${hovered.ratio.value.toFixed(2)} \\pm ${hovered.ratio.err.toFixed(2)}$.`}
            />
          </>
        )}
      </p>
    </div>
  )
}
