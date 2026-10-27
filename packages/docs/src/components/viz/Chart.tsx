import MathLabel from '@site/src/components/MathLabel'
import * as d3 from 'd3'
import React, { type ReactNode, useEffect, useId, useRef } from 'react'

export interface Margin {
  readonly top: number
  readonly right: number
  readonly bottom: number
  readonly left: number
}

export const DEFAULT_MARGIN: Margin = { top: 12, right: 16, bottom: 32, left: 48 }

/** Width of the gutter that holds a rotated axis label. */
const LABEL_GUTTER = 26

export interface InnerSize {
  readonly width: number
  readonly height: number
  /** The clip-path URL of the plot area, for layers that may spill beyond it. */
  readonly clip: string
}

interface ChartProps {
  /** Outer width, usually the measured width of the container. */
  readonly width: number
  readonly height: number
  readonly margin?: Margin
  /** Axis labels, with inline math between dollar signs. */
  readonly xLabel?: string
  readonly yLabel?: string
  readonly children: (inner: InnerSize) => ReactNode
}

/**
 * An SVG chart with its axis labels set in HTML around it, so that they can
 * carry KaTeX. The children are drawn inside the margins and receive the inner
 * size.
 */
export default function Chart({
  width,
  height,
  margin = DEFAULT_MARGIN,
  xLabel,
  yLabel,
  children
}: ChartProps): ReactNode {
  const clipId = `clip-${useId().replace(/[^a-zA-Z0-9]/g, '')}`
  const svgWidth = width - (yLabel === undefined ? 0 : LABEL_GUTTER)
  const inner: InnerSize = {
    width: Math.max(svgWidth - margin.left - margin.right, 10),
    height: Math.max(height - margin.top - margin.bottom, 10),
    clip: `url(#${clipId})`
  }
  return (
    <div className="viz-chart">
      {yLabel === undefined ? null : (
        <div className="viz-chart__ylabel" style={{ height }}>
          <span className="viz-chart__ylabel-text">
            <MathLabel label={yLabel} />
          </span>
        </div>
      )}
      <svg
        className="viz-chart__svg"
        width={svgWidth}
        height={height}
        viewBox={`0 0 ${svgWidth} ${height}`}
      >
        <defs>
          <clipPath id={clipId}>
            <rect x={0} y={0} width={inner.width} height={inner.height} />
          </clipPath>
        </defs>
        <g transform={`translate(${margin.left},${margin.top})`}>{children(inner)}</g>
      </svg>
      {xLabel === undefined ? null : (
        <div className="viz-chart__xlabel" style={{ paddingLeft: margin.left }}>
          <MathLabel label={xLabel} />
        </div>
      )}
    </div>
  )
}

interface AxisProps {
  readonly scale: d3.ScaleLinear<number, number> | d3.ScaleLogarithmic<number, number>
  readonly orientation: 'bottom' | 'left'
  readonly transform?: string
  readonly ticks?: number
  /** Explicit tick positions, which take precedence over the tick count. */
  readonly tickValues?: readonly number[]
  readonly tickFormat?: (value: number) => string
  /** Length of grid lines drawn across the plot, if any. */
  readonly gridLength?: number
}

/** A D3 axis rendered into a group, redrawn whenever its scale changes. */
export function Axis({
  scale,
  orientation,
  transform,
  ticks = 5,
  tickValues,
  tickFormat,
  gridLength
}: AxisProps): ReactNode {
  const ref = useRef<SVGGElement>(null)
  useEffect(() => {
    const group = ref.current
    if (group === null) {
      return
    }
    const axis = orientation === 'bottom' ? d3.axisBottom(scale) : d3.axisLeft(scale)
    axis.ticks(ticks)
    if (tickValues !== undefined) {
      axis.tickValues([...tickValues])
    }
    if (tickFormat !== undefined) {
      axis.tickFormat((value) => tickFormat(value.valueOf()))
    }
    if (gridLength !== undefined) {
      axis.tickSizeInner(-gridLength)
      axis.tickPadding(8)
    }
    const selection = d3.select(group)
    selection.call(axis)
    selection.selectAll('.tick line').attr('class', gridLength === undefined ? null : 'viz-grid')
  }, [scale, orientation, ticks, tickValues, tickFormat, gridLength])
  return <g ref={ref} className="viz-axis" transform={transform} />
}

/** A path through (x, y) pairs already in pixel coordinates. */
export function linePath(points: ReadonlyArray<readonly [number, number]>): string {
  const generator = d3
    .line<readonly [number, number]>()
    .x((point) => point[0])
    .y((point) => point[1])
  return generator(points) ?? ''
}

/** The closed outline of a band between two curves, already in pixel coordinates. */
export function bandPath(
  lower: ReadonlyArray<readonly [number, number]>,
  upper: ReadonlyArray<readonly [number, number]>
): string {
  return `${linePath([...lower, ...[...upper].reverse()])}Z`
}

/** The ticks of a logarithmic axis that fall inside `domain`. */
export function ticksWithin(
  candidates: readonly number[],
  domain: readonly [number, number]
): readonly number[] {
  return candidates.filter((value) => value >= domain[0] && value <= domain[1])
}
