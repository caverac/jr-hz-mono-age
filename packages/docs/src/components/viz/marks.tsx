import MathLabel from '@site/src/components/MathLabel'
import * as d3 from 'd3'
import React, { type ReactNode } from 'react'

/** The shapes of the point markers; the charts tell series apart by shape and fill. */
export type MarkerShape = 'circle' | 'square' | 'triangle' | 'diamond' | 'cross'

/** The dash patterns of the lines. */
export type Dash = 'solid' | 'dashed' | 'dotted' | 'dashdot'

const SYMBOLS: Record<MarkerShape, d3.SymbolType> = {
  circle: d3.symbolCircle,
  square: d3.symbolSquare,
  triangle: d3.symbolTriangle,
  diamond: d3.symbolDiamond,
  cross: d3.symbolTimes
}

/** Marker area in px^2: an 8 px wide circle. */
const MARKER_AREA = 50

/** The radius (px) of the hover target of a marker, larger than the marker. */
const HIT_RADIUS = 8

interface MarkerProps {
  readonly shape: MarkerShape
  readonly x: number
  readonly y: number
  readonly open?: boolean
  /** The color of the marker, such as a halo's (`haloColor`); the foreground color if absent. */
  readonly color?: string
  readonly className?: string
  readonly onMouseEnter?: () => void
  readonly onMouseLeave?: () => void
}

/** One point marker, filled or open, centered on (x, y). */
export function Marker({
  shape,
  x,
  y,
  open = false,
  color,
  className,
  onMouseEnter,
  onMouseLeave
}: MarkerProps): ReactNode {
  const path = d3.symbol(SYMBOLS[shape], MARKER_AREA)() ?? ''
  const outline = shape === 'cross' || open
  const style = shape === 'cross' ? 'viz-marker--stroke' : open ? 'viz-marker--open' : 'viz-marker'
  const mark = (
    <path
      d={path}
      transform={`translate(${x},${y})`}
      className={className === undefined ? style : `${style} ${className}`}
      style={color === undefined ? undefined : outline ? { stroke: color } : { fill: color }}
    />
  )
  if (onMouseEnter === undefined && onMouseLeave === undefined) {
    return mark
  }
  return (
    <g onMouseEnter={onMouseEnter} onMouseLeave={onMouseLeave}>
      {mark}
      <circle cx={x} cy={y} r={HIT_RADIUS} className="viz-hit" />
    </g>
  )
}

/** The number of slots of the categorical palette (`src/css/viz.css`). */
export const SERIES_SLOTS = 8

/**
 * The color of the series `index` (from 0) of a chart, from the categorical palette in its
 * fixed order. A chart with more series than slots must fold or facet them, never cycle.
 */
export function seriesColor(index: number): string {
  if (!Number.isInteger(index) || index < 0 || index >= SERIES_SLOTS) {
    throw new RangeError(`The palette has ${SERIES_SLOTS} slots, not a slot ${index + 1}.`)
  }
  return `var(--viz-series-${index + 1})`
}

/** The color key of a halo or a series, set before its name in buttons and readouts. */
export function Swatch({ color }: { readonly color: string }): ReactNode {
  return <span className="viz-swatch" style={{ background: color }} aria-hidden="true" />
}

/** The CSS class of a line with the dash pattern `dash`. */
export function dashClass(dash: Dash): string {
  return dash === 'solid' ? 'viz-line' : `viz-line viz-line--${dash}`
}

export interface LegendEntry {
  /** The label, with inline math between dollar signs. */
  readonly label: string
  readonly dash?: Dash
  readonly marker?: MarkerShape
  readonly open?: boolean
  /** A shaded band, such as a tolerance or a percentile range. */
  readonly band?: boolean
  /** The color of the line and the marker, as in the chart; the foreground color if absent. */
  readonly color?: string
}

/** The sample of one legend entry: its band, line and marker, in a small SVG. */
function Sample({ entry }: { readonly entry: LegendEntry }): ReactNode {
  return (
    <svg className="viz-legend__sample" width={30} height={14} viewBox="0 0 30 14">
      {entry.band === true ? (
        <rect x={0} y={2} width={30} height={10} className="viz-band" />
      ) : null}
      {entry.dash === undefined ? null : (
        <line
          x1={0}
          x2={30}
          y1={7}
          y2={7}
          className={dashClass(entry.dash)}
          style={entry.color === undefined ? undefined : { stroke: entry.color }}
        />
      )}
      {entry.marker === undefined ? null : (
        <Marker shape={entry.marker} x={15} y={7} open={entry.open} color={entry.color} />
      )}
    </svg>
  )
}

/** A legend set in HTML under a chart, so that its labels can carry KaTeX. */
export function Legend({ entries }: { readonly entries: readonly LegendEntry[] }): ReactNode {
  return (
    <ul className="viz-legend">
      {entries.map((entry) => (
        <li key={entry.label} className="viz-legend__item">
          <Sample entry={entry} />
          <MathLabel label={entry.label} />
        </li>
      ))}
    </ul>
  )
}
