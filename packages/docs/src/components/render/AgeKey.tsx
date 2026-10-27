import * as d3 from 'd3'
import React, { type ReactNode, useEffect, useId, useMemo, useRef } from 'react'

/** The ages (Gyr) that the key spans. */
export const AGE_DOMAIN: readonly [number, number] = [0, 13.5]

/** The smallest range of ages (Gyr) the brush selects. */
const MIN_SPAN_GYR = 0.1

const MARGIN = { left: 12, right: 12 }
const BAR_TOP = 18
const BAR_HEIGHT = 12
const HEIGHT = 50

type Range = readonly [number, number]

interface AgeKeyProps {
  readonly width: number
  /** The ages (Gyr) at which `colors` apply; the color is interpolated between them. */
  readonly stops: readonly number[]
  readonly colors: readonly string[]
  /** The selected range of ages; the key dims the ages outside it. */
  readonly range?: Range
  /** Called with the range of ages brushed on the key; the whole domain when the brush is cleared. */
  readonly onRange?: (range: Range) => void
}

/** Whether `range` spans the whole key. */
export function isWholeDomain(range: Range): boolean {
  return range[0] <= AGE_DOMAIN[0] && range[1] >= AGE_DOMAIN[1]
}

/** Round `age` (Gyr) to 0.1 Gyr. */
function tenth(age: number): number {
  return Math.round(age * 10) / 10
}

/**
 * A color bar of stellar age with its axis in Gyr, drawn with d3. With `onRange`, the
 * bar is also a brush: dragging across it selects a range of ages.
 */
export default function AgeKey({ width, stops, colors, range, onRange }: AgeKeyProps): ReactNode {
  const axisRef = useRef<SVGGElement>(null)
  const brushRef = useRef<SVGGElement>(null)
  const brush = useRef<d3.BrushBehavior<unknown> | null>(null)
  const gradientId = `age-key-${useId().replace(/[^a-zA-Z0-9]/g, '')}`
  const [lo, hi] = AGE_DOMAIN
  const x = useMemo(
    () =>
      d3
        .scaleLinear()
        .domain(AGE_DOMAIN)
        .range([MARGIN.left, width - MARGIN.right])
        .clamp(true),
    [width]
  )

  useEffect(() => {
    if (axisRef.current !== null) {
      const ticks = Math.max(2, Math.min(14, Math.floor(width / 55)))
      d3.select(axisRef.current).call(d3.axisBottom(x).ticks(ticks).tickSizeOuter(0))
    }
  }, [x, width])

  useEffect(() => {
    const node = brushRef.current
    if (node === null || onRange === undefined) {
      return undefined
    }
    const behavior = d3
      .brushX<unknown>()
      .extent([
        [x(lo), BAR_TOP - 3],
        [x(hi), BAR_TOP + BAR_HEIGHT + 3]
      ])
      .on('end', (event: d3.D3BrushEvent<unknown>) => {
        const source: unknown = event.sourceEvent
        if (source === null || source === undefined) {
          return
        }
        const selection = event.selection
        if (
          selection === null ||
          typeof selection[0] !== 'number' ||
          typeof selection[1] !== 'number'
        ) {
          onRange(AGE_DOMAIN)
          return
        }
        const from = tenth(x.invert(selection[0]))
        onRange([from, Math.max(tenth(x.invert(selection[1])), from + MIN_SPAN_GYR)])
      })
    brush.current = behavior
    const group = d3.select(node).call(behavior)
    return () => {
      group.on('.brush', null)
      group.selectAll('*').remove()
      brush.current = null
    }
  }, [x, lo, hi, onRange])

  useEffect(() => {
    const node = brushRef.current
    if (node === null || brush.current === null || range === undefined) {
      return
    }
    d3.select(node).call(
      brush.current.move,
      isWholeDomain(range) ? null : [x(range[0]), x(range[1])]
    )
  }, [range, x])

  return (
    <svg
      className="age-key"
      width={width}
      height={HEIGHT}
      role="img"
      aria-label="Color key of stellar age, in Gyr"
    >
      <defs>
        <linearGradient
          id={gradientId}
          gradientUnits="userSpaceOnUse"
          x1={x(lo)}
          x2={x(hi)}
          y1={0}
          y2={0}
        >
          {stops.map((stop, i) => (
            <stop key={stop} offset={(stop - lo) / (hi - lo)} stopColor={colors[i]} />
          ))}
        </linearGradient>
      </defs>
      <text x={x(lo)} y={BAR_TOP - 6}>
        age (Gyr){onRange === undefined ? '' : ', drag across the bar to choose'}
      </text>
      <rect
        x={x(lo)}
        y={BAR_TOP}
        width={x(hi) - x(lo)}
        height={BAR_HEIGHT}
        rx={2}
        fill={`url(#${gradientId})`}
      />
      {range === undefined || isWholeDomain(range) ? null : (
        <g fill="black" fillOpacity={0.72} pointerEvents="none">
          <rect x={x(lo)} y={BAR_TOP} width={x(range[0]) - x(lo)} height={BAR_HEIGHT} />
          <rect x={x(range[1])} y={BAR_TOP} width={x(hi) - x(range[1])} height={BAR_HEIGHT} />
        </g>
      )}
      <g ref={axisRef} transform={`translate(0, ${BAR_TOP + BAR_HEIGHT})`} />
      {onRange === undefined ? null : <g ref={brushRef} className="age-key__brush" />}
    </svg>
  )
}
