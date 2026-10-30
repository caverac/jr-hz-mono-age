import MathLabel from '@site/src/components/MathLabel'
import { haloColor } from '@site/src/components/render/halos'
import { fadeClass, HaloPicker, Pending } from '@site/src/components/status/controls'
import type { HaloRecord, StatusData } from '@site/src/components/status/data'
import { ageRange } from '@site/src/components/status/metrics'
import useStatusData from '@site/src/components/status/useStatusData'
import Chart, { Axis, linePath, ticksWithin } from '@site/src/components/viz/Chart'
import { useElementWidth, useRevealed } from '@site/src/components/viz/hooks'
import { Legend, type LegendEntry, Marker } from '@site/src/components/viz/marks'
import * as d3 from 'd3'
import React, { type ReactNode, useState } from 'react'

const HEIGHT = 360
const SIGMA_TICKS = [5, 7, 10, 15, 20, 30, 40, 50, 70, 100, 150]
/** The ticks of an x axis narrower than NARROW px, where the full set would collide. */
const SPARSE_SIGMA_TICKS = [5, 10, 20, 50, 100]
const NARROW = 380
const ETA_TICKS = [1, 1.25, 1.5, 1.75, 2]
const Y_DOMAIN: readonly [number, number] = [0.9, 2.1]
/** The range of ages (Gyr) over which each halo's measured eta is fitted. */
const ETA_RANGE: readonly [number, number] = [0, 10]

/** One halo's measured eta over ETA_RANGE, its error, and the sigma_z (km/s) of the populations it rests on. */
interface Measured {
  readonly eta: number
  readonly err: number
  readonly sigmaLo: number
  readonly sigmaHi: number
  readonly sigmaCenter: number
}

function measuredEta(halo: HaloRecord): Measured | null {
  const range = ageRange(halo, ...ETA_RANGE)
  const sigmas = halo.bins
    .filter((b) => b.ageLo >= ETA_RANGE[0] && b.ageHi <= ETA_RANGE[1])
    .map((b) => b.sigmaZ.value)
    .filter((s) => Number.isFinite(s) && s > 0)
  if (range === undefined || sigmas.length === 0) {
    return null
  }
  return {
    eta: range.eta.value,
    err: range.eta.err,
    sigmaLo: d3.min(sigmas) ?? Number.NaN,
    sigmaHi: d3.max(sigmas) ?? Number.NaN,
    sigmaCenter: Math.exp(d3.mean(sigmas.map(Math.log)) ?? 0)
  }
}

/** The points of a curve of one halo where its local slope is finite, as runs between gaps. */
function runs(halo: HaloRecord): readonly (readonly (readonly [number, number])[])[] {
  const out: (readonly [number, number])[][] = [[]]
  halo.tracerCurve.sigmaZ.forEach((s, i) => {
    const eta = halo.tracerCurve.eta[i]
    if (Number.isFinite(eta)) {
      out[out.length - 1].push([s, eta])
    } else if (out[out.length - 1].length > 0) {
      out.push([])
    }
  })
  return out.filter((run) => run.length > 1)
}

function Plot({
  data,
  width,
  selected,
  revealed
}: {
  readonly data: StatusData
  readonly width: number
  readonly selected: string | null
  readonly revealed: boolean
}): ReactNode {
  const sigmas = data.halos.flatMap((h) => h.tracerCurve.sigmaZ)
  const xDomain: readonly [number, number] = [d3.min(sigmas) ?? 5, d3.max(sigmas) ?? 150]
  return (
    <Chart width={width} height={HEIGHT} xLabel="$\sigma_z$ (km/s)" yLabel="$\eta$">
      {(inner) => {
        const x = d3.scaleLog().domain(xDomain).range([0, inner.width])
        const y = d3.scaleLinear().domain(Y_DOMAIN).range([inner.height, 0])
        const level = (eta: number) => linePath(xDomain.map((s) => [x(s), y(eta)] as const))
        return (
          <g className={revealed ? 'viz-reveal viz-reveal--in' : 'viz-reveal'}>
            <Axis
              scale={y}
              orientation="left"
              tickValues={ticksWithin(ETA_TICKS, Y_DOMAIN)}
              tickFormat={(v) => String(v)}
              gridLength={inner.width}
            />
            <Axis
              scale={x}
              orientation="bottom"
              transform={`translate(0,${inner.height})`}
              tickValues={ticksWithin(
                inner.width < NARROW ? SPARSE_SIGMA_TICKS : SIGMA_TICKS,
                xDomain
              )}
              tickFormat={(v) => String(v)}
              gridLength={inner.height}
            />
            <g clipPath={inner.clip} className="viz-fadein">
              <path d={level(1)} className="viz-guide viz-line--dotted" />
              <path d={level(2)} className="viz-guide viz-line--dashed" />
            </g>
            <g clipPath={inner.clip}>
              {data.halos.map((halo, hi) => {
                const color = haloColor(halo.label)
                const measured = measuredEta(halo)
                return (
                  <g key={halo.label} className={fadeClass(selected, halo.label)}>
                    {runs(halo).map((run, ri) => (
                      <path
                        key={ri}
                        d={linePath(run.map(([s, eta]) => [x(s), y(eta)] as const))}
                        pathLength={1}
                        className="viz-line viz-draw"
                        style={{ stroke: color, animationDelay: `${200 + hi * 90}ms` }}
                      />
                    ))}
                    {measured === null ? null : (
                      <g className="viz-pop" style={{ animationDelay: `${700 + hi * 60}ms` }}>
                        <line
                          x1={x(measured.sigmaLo)}
                          x2={x(measured.sigmaHi)}
                          y1={y(measured.eta)}
                          y2={y(measured.eta)}
                          className="viz-line viz-line--thin"
                          style={{ stroke: color }}
                        />
                        <line
                          x1={x(measured.sigmaCenter)}
                          x2={x(measured.sigmaCenter)}
                          y1={y(measured.eta - measured.err)}
                          y2={y(measured.eta + measured.err)}
                          className="viz-line viz-line--thin"
                          style={{ stroke: color }}
                        />
                        <Marker
                          shape="circle"
                          color={color}
                          x={x(measured.sigmaCenter)}
                          y={y(measured.eta)}
                        />
                      </g>
                    )}
                  </g>
                )
              })}
            </g>
          </g>
        )
      }}
    </Chart>
  )
}

const ENTRIES: readonly LegendEntry[] = [
  { label: 'local slope of the tracer', dash: 'solid' },
  { label: 'measured $\\eta$ over 0-10 Gyr', marker: 'circle' },
  { label: '$\\eta = 1$', dash: 'dotted' },
  { label: '$\\eta = 2$', dash: 'dashed' }
]

function Readout({
  data,
  selected
}: {
  readonly data: StatusData
  readonly selected: string | null
}): ReactNode {
  const halo = data.halos.find((h) => h.label === selected)
  if (halo === undefined) {
    return (
      <p className="viz-note">Pick a halo to read where its tracer has no exponential scale.</p>
    )
  }
  const { sigmaZ, hz } = halo.tracerCurve
  const first = sigmaZ.find((_, i) => !Number.isFinite(hz[i]))
  const measured = measuredEta(halo)
  const etaText =
    measured === null
      ? ''
      : `measured $\\eta = ${measured.eta.toFixed(2)} \\pm ${measured.err.toFixed(2)}$; `
  const scaleText =
    first === undefined
      ? 'the window finds a scale at every dispersion of the grid.'
      : `the window finds no scale above $\\sigma_z = ${first.toFixed(0)}$ km/s.`
  return (
    <p className="viz-note">
      <MathLabel label={`${halo.label}: ${etaText}${scaleText}`} />
    </p>
  )
}

/**
 * The local slope eta = d ln h_Z / d ln sigma_z of the tracer of each halo's fitted potential
 * at the median radius of its annulus, against sigma_z, with the eta fitted to the heights and
 * dispersions of its populations over 0-10 Gyr, drawn across the dispersions they span. A gap marks dispersions at which
 * the self-consistent window finds no exponential scale.
 */
export default function EtaChart(): ReactNode {
  const [ref, width] = useElementWidth<HTMLDivElement>()
  const revealed = useRevealed(ref)
  const { value, error } = useStatusData()
  const [selected, setSelected] = useState<string | null>(null)
  if (value === null) {
    return (
      <div className="viz" ref={ref}>
        <Pending error={error} />
      </div>
    )
  }
  return (
    <div className="viz" ref={ref}>
      <Plot data={value} width={width} selected={selected} revealed={revealed} />
      <Legend entries={ENTRIES} />
      <div className="viz-controls">
        <HaloPicker selected={selected} onSelect={setSelected} />
      </div>
      <Readout data={value} selected={selected} />
    </div>
  )
}
