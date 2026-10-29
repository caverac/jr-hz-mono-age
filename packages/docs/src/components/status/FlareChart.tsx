import MathLabel from '@site/src/components/MathLabel'
import { haloColor } from '@site/src/components/render/halos'
import { fadeClass, HaloPicker, Pending } from '@site/src/components/status/controls'
import type { StatusData } from '@site/src/components/status/data'
import type { SumRange } from '@site/src/components/status/tracerSum'
import useStatusData from '@site/src/components/status/useStatusData'
import Chart, { Axis, linePath } from '@site/src/components/viz/Chart'
import { useElementWidth, useRevealed } from '@site/src/components/viz/hooks'
import { Legend, type LegendEntry, Marker, Swatch } from '@site/src/components/viz/marks'
import * as d3 from 'd3'
import React, { type ReactNode, useState } from 'react'

const HEIGHT_TOP = 300
const HEIGHT_BOTTOM = 200
/** The band of relative agreement drawn around a ratio of 1. */
const TOLERANCE = 0.1

/** The center (kpc) of a radial range, where its marker sits. */
function center(ring: SumRange): number {
  return 0.5 * (ring.rLo + ring.rHi)
}

interface Hovered {
  readonly halo: string
  readonly ring: SumRange
}

interface PanelProps {
  readonly data: StatusData
  readonly width: number
  readonly xDomain: readonly [number, number]
  readonly selected: string | null
  readonly revealed: boolean
  readonly onHover: (hovered: Hovered | null) => void
}

/** The flare: the measured h_Z of each ring, and the line of the sum of tracers through the rings. */
function HeightPanel({ data, width, xDomain, selected, revealed, onHover }: PanelProps): ReactNode {
  const heights = data.halos.flatMap((h) =>
    h.tracerSum.rings.flatMap((r) => [r.measured.scaled.value, r.mixture.scaled.value])
  )
  const yDomain: readonly [number, number] = [0, (d3.max(heights) ?? 3) * 1.12]
  return (
    <Chart width={width} height={HEIGHT_TOP} yLabel="$h_Z$ (kpc)">
      {(inner) => {
        const x = d3.scaleLinear().domain(xDomain).range([0, inner.width])
        const y = d3.scaleLinear().domain(yDomain).range([inner.height, 0])
        return (
          <g className={revealed ? 'viz-reveal viz-reveal--in' : 'viz-reveal'}>
            <Axis scale={y} orientation="left" ticks={5} gridLength={inner.width} />
            <Axis
              scale={x}
              orientation="bottom"
              transform={`translate(0,${inner.height})`}
              ticks={8}
              gridLength={inner.height}
            />
            {data.halos.map((halo, hi) => {
              const color = haloColor(halo.label)
              const rings = halo.tracerSum.rings
              return (
                <g key={halo.label} className={fadeClass(selected, halo.label)}>
                  <path
                    d={linePath(
                      rings.map((r) => [x(center(r)), y(r.mixture.scaled.value)] as const)
                    )}
                    pathLength={1}
                    className="viz-line viz-draw"
                    style={{ stroke: color, animationDelay: `${hi * 120}ms` }}
                  />
                  {rings.map((r, ri) => (
                    <g
                      key={r.rLo}
                      className="viz-pop"
                      style={{ animationDelay: `${hi * 120 + ri * 110}ms` }}
                    >
                      <line
                        x1={x(r.rLo)}
                        x2={x(r.rHi)}
                        y1={y(r.measured.scaled.value)}
                        y2={y(r.measured.scaled.value)}
                        className="viz-line viz-line--thin"
                        style={{ stroke: color }}
                      />
                      <Marker
                        shape="circle"
                        open
                        color={color}
                        x={x(center(r))}
                        y={y(r.measured.scaled.value)}
                        onMouseEnter={() => onHover({ halo: halo.label, ring: r })}
                        onMouseLeave={() => onHover(null)}
                      />
                    </g>
                  ))}
                </g>
              )
            })}
          </g>
        )
      }}
    </Chart>
  )
}

/** The sum of tracers over the measured h_Z of each ring. */
function RatioPanel({ data, width, xDomain, selected, revealed, onHover }: PanelProps): ReactNode {
  const values = data.halos.flatMap((h) =>
    h.tracerSum.rings.map((r) => r.ratioMixture.scaled.value)
  )
  const spread = Math.max(TOLERANCE * 1.6, ...values.map((v) => Math.abs(v - 1) * 1.15))
  const yDomain: readonly [number, number] = [1 - spread, 1 + spread]
  return (
    <Chart width={width} height={HEIGHT_BOTTOM} xLabel="$R$ (kpc)" yLabel="$h_\mathrm{sum}/h_Z$">
      {(inner) => {
        const x = d3.scaleLinear().domain(xDomain).range([0, inner.width])
        const y = d3.scaleLinear().domain(yDomain).range([inner.height, 0])
        return (
          <g className={revealed ? 'viz-reveal viz-reveal--in' : 'viz-reveal'}>
            <rect
              x={0}
              width={inner.width}
              y={y(1 + TOLERANCE)}
              height={y(1 - TOLERANCE) - y(1 + TOLERANCE)}
              className="viz-band"
            />
            <Axis scale={y} orientation="left" ticks={4} gridLength={inner.width} />
            <Axis
              scale={x}
              orientation="bottom"
              transform={`translate(0,${inner.height})`}
              ticks={8}
              gridLength={inner.height}
            />
            <line x1={0} x2={inner.width} y1={y(1)} y2={y(1)} className="viz-line viz-line--thin" />
            {data.halos.map((halo, hi) => {
              const color = haloColor(halo.label)
              const rings = halo.tracerSum.rings
              return (
                <g key={halo.label} className={fadeClass(selected, halo.label)}>
                  <path
                    d={linePath(
                      rings.map((r) => [x(center(r)), y(r.ratioMixture.scaled.value)] as const)
                    )}
                    pathLength={1}
                    className="viz-line viz-line--thin viz-draw"
                    style={{ stroke: color, animationDelay: `${600 + hi * 120}ms` }}
                  />
                  {rings.map((r, ri) => {
                    const ratio = r.ratioMixture.scaled
                    return (
                      <g
                        key={r.rLo}
                        className="viz-pop"
                        style={{ animationDelay: `${600 + hi * 120 + ri * 110}ms` }}
                      >
                        <line
                          x1={x(center(r))}
                          x2={x(center(r))}
                          y1={y(ratio.value - ratio.err)}
                          y2={y(ratio.value + ratio.err)}
                          className="viz-line viz-line--thin"
                          style={{ stroke: color }}
                        />
                        <Marker
                          shape="circle"
                          color={color}
                          x={x(center(r))}
                          y={y(ratio.value)}
                          onMouseEnter={() => onHover({ halo: halo.label, ring: r })}
                          onMouseLeave={() => onHover(null)}
                        />
                      </g>
                    )
                  })}
                </g>
              )
            })}
          </g>
        )
      }}
    </Chart>
  )
}

function Readout({ hovered }: { readonly hovered: Hovered | null }): ReactNode {
  if (hovered === null) {
    return <p className="viz-note">Point at a ring to read its heights.</p>
  }
  const { halo, ring } = hovered
  const m = ring.measured.scaled
  const r = ring.ratioMixture.scaled
  return (
    <p className="viz-note">
      <Swatch color={haloColor(halo)} />
      <MathLabel
        label={`${halo}, ${ring.rLo}-${ring.rHi} kpc: $h_Z = ${m.value.toFixed(2)} \\pm ${m.err.toFixed(2)}$ kpc, sum of tracers $${ring.mixture.scaled.value.toFixed(2)}$ kpc, a ratio of $${r.value.toFixed(2)} \\pm ${r.err.toFixed(2)}$.`}
      />
    </p>
  )
}

const ENTRIES: readonly LegendEntry[] = [
  { label: 'measured', marker: 'circle', open: true },
  { label: 'sum of tracers', dash: 'solid' },
  { label: `$|h_\\mathrm{sum}/h_Z - 1| \\le ${TOLERANCE}$`, band: true }
]

/**
 * The whole-disk h_Z of every halo in radial ranges, in the self-consistent window (top), and
 * the h_Z of the sum of the isothermal tracers of its rings and ages over it (bottom), on a
 * shared radius.
 */
export default function FlareChart(): ReactNode {
  const [ref, width] = useElementWidth<HTMLDivElement>()
  const revealed = useRevealed(ref)
  const { value, error } = useStatusData()
  const [selected, setSelected] = useState<string | null>(null)
  const [hovered, setHovered] = useState<Hovered | null>(null)
  if (value === null) {
    return (
      <div className="viz" ref={ref}>
        <Pending error={error} />
      </div>
    )
  }
  const rings = value.halos.flatMap((h) => h.tracerSum.rings)
  const xDomain: readonly [number, number] = [
    d3.min(rings, (r) => r.rLo) ?? 0,
    d3.max(rings, (r) => r.rHi) ?? 20
  ]
  const panel = {
    data: value,
    width,
    xDomain,
    selected,
    revealed,
    onHover: setHovered
  }
  return (
    <div className="viz" ref={ref}>
      <HeightPanel {...panel} />
      <RatioPanel {...panel} />
      <Legend entries={ENTRIES} />
      <div className="viz-controls">
        <HaloPicker selected={selected} onSelect={setSelected} />
      </div>
      <Readout hovered={hovered} />
    </div>
  )
}
