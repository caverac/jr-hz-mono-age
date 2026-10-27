import {
  Chips,
  fadeClass,
  HaloPicker,
  Pending,
  toggled
} from '@site/src/components/status/controls'
import type { Injection, Spread, StatusData } from '@site/src/components/status/data'
import { FAMILIES } from '@site/src/components/status/families'
import { family, windowPoints } from '@site/src/components/status/metrics'
import useStatusData from '@site/src/components/status/useStatusData'
import Chart, { Axis, bandPath, linePath, ticksWithin } from '@site/src/components/viz/Chart'
import { useElementWidth } from '@site/src/components/viz/hooks'
import {
  dashClass,
  Legend,
  Marker,
  SMALL_MARKER_AREA,
  type Dash
} from '@site/src/components/viz/marks'
import * as d3 from 'd3'
import React, { type ReactNode, useState } from 'react'

const TOP_HEIGHT = 320
const BOTTOM_HEIGHT = 200
const LOG_TICKS = [0.1, 0.2, 0.3, 0.5, 0.7, 1, 1.5, 2, 3, 5, 7, 10]

type Scale = d3.ScaleLogarithmic<number, number> | d3.ScaleLinear<number, number>

interface CurveProps {
  readonly injection: Injection
  readonly name: string
  readonly dash: Dash
  readonly pick: (spread: { fixed: Spread; scaled: Spread }) => Spread
  readonly x: Scale
  readonly y: Scale
}

/** The median of one family and the band of its 16th to 84th percentiles over the draws. */
function Curve({ injection, name, dash, pick, x, y }: CurveProps): ReactNode {
  const rows = family(injection, name)
    .map((p) => ({ x: x(p.reference / injection.zLim), s: pick(p) }))
    .filter(({ s }) => Number.isFinite(s.median))
  return (
    <g>
      <path
        d={bandPath(
          rows.map((r) => [r.x, y(r.s.lo)] as const),
          rows.map((r) => [r.x, y(r.s.hi)] as const)
        )}
        className="viz-band"
      />
      <path
        d={linePath(rows.map((r) => [r.x, y(r.s.median)] as const))}
        className={dashClass(dash)}
      />
      {rows.map((r) => (
        <Marker key={r.x} shape="square" open x={r.x} y={y(r.s.median)} size={SMALL_MARKER_AREA} />
      ))}
    </g>
  )
}

interface PlotProps {
  readonly data: StatusData
  readonly width: number
  readonly shown: ReadonlySet<string>
  readonly selected: string | null
}

function Plots({ data, width, shown, selected }: PlotProps): ReactNode {
  const { injection } = data
  const halos = windowPoints(data).filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y))
  const xs = [
    ...injection.points.map((p) => p.reference / injection.zLim),
    ...halos.map((p) => p.x)
  ]
  const xDomain: readonly [number, number] = [(d3.min(xs) ?? 0.1) / 1.25, (d3.max(xs) ?? 10) * 1.25]
  const fixedMedians = injection.points.map((p) => p.fixed.median).filter(Number.isFinite)
  const ys = [...fixedMedians, ...halos.map((p) => p.y)]
  const topDomain: readonly [number, number] = [(d3.min(ys) ?? 0.5) / 1.2, (d3.max(ys) ?? 2) * 1.2]
  const scaledEnds = injection.points.flatMap((p) => [p.scaled.lo, p.scaled.hi])
  const half = Math.max(...scaledEnds.map((v) => Math.abs(v - 1))) * 1.2
  const bottomDomain: readonly [number, number] = [1 - half, 1 + half]
  const families = FAMILIES.filter(([name]) => shown.has(name))
  return (
    <>
      <Chart width={width} height={TOP_HEIGHT} yLabel="$h_{Z_\mathrm{lim}}/h_Z$">
        {(inner) => {
          const x = d3.scaleLog().domain(xDomain).range([0, inner.width])
          const y = d3.scaleLog().domain(topDomain).range([inner.height, 0])
          return (
            <>
              <Axis
                scale={y}
                orientation="left"
                tickValues={ticksWithin(LOG_TICKS, topDomain)}
                tickFormat={(v) => String(v)}
                gridLength={inner.width}
              />
              <Axis
                scale={x}
                orientation="bottom"
                transform={`translate(0,${inner.height})`}
                tickValues={ticksWithin(LOG_TICKS, xDomain)}
                tickFormat={() => ''}
                gridLength={inner.height}
              />
              <line
                x1={0}
                x2={inner.width}
                y1={y(1)}
                y2={y(1)}
                className="viz-line viz-line--thin"
              />
              <g clipPath={inner.clip}>
                {families.map(([name, , dash]) => (
                  <Curve
                    key={name}
                    injection={injection}
                    name={name}
                    dash={dash}
                    pick={(p) => p.fixed}
                    x={x}
                    y={y}
                  />
                ))}
              </g>
              {halos.map((p, i) => (
                <Marker
                  key={i}
                  shape="circle"
                  x={x(p.x)}
                  y={y(p.y)}
                  size={SMALL_MARKER_AREA}
                  className={fadeClass(selected, p.halo)}
                />
              ))}
            </>
          )
        }}
      </Chart>
      <Chart
        width={width}
        height={BOTTOM_HEIGHT}
        xLabel="$h_Z/Z_\mathrm{lim}$"
        yLabel="$h_Z/h_\mathrm{ref}$"
      >
        {(inner) => {
          const x = d3.scaleLog().domain(xDomain).range([0, inner.width])
          const y = d3.scaleLinear().domain(bottomDomain).range([inner.height, 0])
          return (
            <>
              <Axis scale={y} orientation="left" ticks={4} gridLength={inner.width} />
              <Axis
                scale={x}
                orientation="bottom"
                transform={`translate(0,${inner.height})`}
                tickValues={ticksWithin(LOG_TICKS, xDomain)}
                tickFormat={(v) => String(v)}
                gridLength={inner.height}
              />
              <line
                x1={0}
                x2={inner.width}
                y1={y(1)}
                y2={y(1)}
                className="viz-line viz-line--thin"
              />
              <g clipPath={inner.clip}>
                {families.map(([name, , dash]) => (
                  <Curve
                    key={name}
                    injection={injection}
                    name={name}
                    dash={dash}
                    pick={(p) => p.scaled}
                    x={x}
                    y={y}
                  />
                ))}
              </g>
            </>
          )
        }}
      </Chart>
    </>
  )
}

/**
 * The fixed-window h_Z over the self-consistent h_Z against h_Z / Z_lim: the injected
 * profiles of known shape (median and 16th to 84th percentiles over the draws) and the
 * mono-age populations of the seven halos. Below, the self-consistent fit of the injected
 * draws over the exact scale of their profile.
 */
export default function WindowBiasChart(): ReactNode {
  const [ref, width] = useElementWidth<HTMLDivElement>()
  const { value, error } = useStatusData()
  const [shown, setShown] = useState<ReadonlySet<string>>(new Set(FAMILIES.map(([name]) => name)))
  const [selected, setSelected] = useState<string | null>(null)
  return (
    <div className="viz" ref={ref}>
      {value === null ? (
        <Pending error={error} />
      ) : (
        <>
          <Plots data={value} width={width} shown={shown} selected={selected} />
          <Legend
            entries={[
              ...FAMILIES.filter(([name]) => shown.has(name)).map(([, label, dash]) => ({
                label,
                dash,
                marker: 'square' as const,
                open: true
              })),
              { label: '16th to 84th percentile', band: true },
              { label: 'Auriga populations', marker: 'circle' }
            ]}
          />
          <div className="viz-controls">
            <Chips
              label="Profiles"
              options={FAMILIES.map(([name, label]) => [name, label] as const)}
              isPressed={(name) => shown.has(name)}
              onToggle={(name) => setShown((current) => toggled(current, name))}
            />
            <HaloPicker selected={selected} onSelect={setSelected} />
          </div>
        </>
      )}
    </div>
  )
}
