import { Pending } from '@site/src/components/status/controls'
import type { Estimate, StatusData } from '@site/src/components/status/data'
import useStatusData from '@site/src/components/status/useStatusData'
import Chart, { Axis, DEFAULT_MARGIN } from '@site/src/components/viz/Chart'
import { useElementWidth } from '@site/src/components/viz/hooks'
import { Legend, Marker, type MarkerShape } from '@site/src/components/viz/marks'
import * as d3 from 'd3'
import React, { type ReactNode } from 'react'

const ROW_HEIGHT = 30
/** Vertical distance (px) between the estimates of the series within the row of a halo. */
const SERIES_GAP = 7
const MARGIN = { ...DEFAULT_MARGIN, top: 24 }

/** One series of estimates, drawn with its own marker in every row. */
export interface RowSeries {
  readonly key: string
  /** The legend label, with inline math between dollar signs. */
  readonly label: string
  readonly marker: MarkerShape
  readonly open: boolean
}

/** The estimates of one halo, by the key of their series. */
export interface HaloRow {
  readonly halo: string
  readonly values: Readonly<Record<string, Estimate>>
}

/** The rows of one range of ages. */
export interface RowPanel {
  readonly label: string
  readonly rows: readonly HaloRow[]
}

interface PanelProps {
  readonly panel: RowPanel
  readonly series: readonly RowSeries[]
  readonly domain: readonly [number, number]
  readonly width: number
  /** The label of the shared x axis, drawn under the last panel only. */
  readonly xLabel?: string
  readonly reference: number
}

/** The estimates of every halo over one range of ages, with their 1-sigma error bars. */
function Panel({ panel, series, domain, width, xLabel, reference }: PanelProps): ReactNode {
  const { rows } = panel
  const height = MARGIN.top + MARGIN.bottom + ROW_HEIGHT * rows.length
  return (
    <Chart width={width} height={height} margin={MARGIN} xLabel={xLabel}>
      {(inner) => {
        const x = d3.scaleLinear().domain(domain).range([0, inner.width]).nice()
        const y = d3
          .scaleLinear()
          .domain([-0.5, rows.length - 0.5])
          .range([0, inner.height])
        const offset = (i: number) => (i - (series.length - 1) / 2) * SERIES_GAP
        return (
          <>
            <Axis
              scale={x}
              orientation="bottom"
              transform={`translate(0,${inner.height})`}
              ticks={6}
              tickFormat={xLabel === undefined ? () => '' : (v) => String(v)}
              gridLength={inner.height}
            />
            <Axis
              scale={y}
              orientation="left"
              tickValues={rows.map((_, i) => i)}
              tickFormat={(i) => rows[i]?.halo ?? ''}
            />
            <line
              x1={x(reference)}
              x2={x(reference)}
              y1={0}
              y2={inner.height}
              className="viz-line viz-line--thin"
            />
            <text x={inner.width} y={-8} textAnchor="end" className="viz-text">
              {panel.label}
            </text>
            {rows.map((row, i) =>
              series.map((s, j) => {
                const estimate = row.values[s.key]
                const at = y(i) + offset(j)
                return estimate === undefined ? null : (
                  <g key={`${row.halo}-${s.key}`}>
                    <line
                      x1={x(estimate.value - estimate.err)}
                      x2={x(estimate.value + estimate.err)}
                      y1={at}
                      y2={at}
                      className="viz-line viz-line--thin"
                    />
                    <Marker shape={s.marker} open={s.open} x={x(estimate.value)} y={at} />
                  </g>
                )
              })
            )}
          </>
        )
      }}
    </Chart>
  )
}

interface HaloRowsChartProps {
  /** The panels, one per range of ages, built from the records. */
  readonly build: (data: StatusData) => readonly RowPanel[]
  readonly series: readonly RowSeries[]
  readonly xLabel: string
  /** The x of the vertical reference line. */
  readonly reference?: number
}

/**
 * Estimates of every halo, one row per halo and one panel per range of ages, stacked on a
 * shared x axis.
 */
export default function HaloRowsChart({
  build,
  series,
  xLabel,
  reference = 0
}: HaloRowsChartProps): ReactNode {
  const [ref, width] = useElementWidth<HTMLDivElement>()
  const { value, error } = useStatusData()
  if (value === null) {
    return (
      <div className="viz" ref={ref}>
        <Pending error={error} />
      </div>
    )
  }
  const panels = build(value)
  const ends = panels.flatMap((panel) =>
    panel.rows.flatMap((row) =>
      Object.values(row.values).flatMap((e) => [e.value - e.err, e.value + e.err])
    )
  )
  const domain: readonly [number, number] = [
    Math.min(reference, d3.min(ends) ?? reference),
    Math.max(reference, d3.max(ends) ?? reference)
  ]
  return (
    <div className="viz" ref={ref}>
      {panels.map((panel, i) => (
        <Panel
          key={panel.label}
          panel={panel}
          series={series}
          domain={domain}
          width={width}
          xLabel={i === panels.length - 1 ? xLabel : undefined}
          reference={reference}
        />
      ))}
      <Legend entries={series} />
    </div>
  )
}
