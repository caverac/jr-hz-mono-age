import HaloChips, { DEFAULT_HALO, haloOf } from '@site/src/components/concepts/HaloChips'
import { type PopulationFit, useConcepts } from '@site/src/components/concepts/record'
import MathLabel from '@site/src/components/MathLabel'
import { Chips, Pending } from '@site/src/components/status/controls'
import Chart, { Axis, linePath, ticksWithin } from '@site/src/components/viz/Chart'
import { useElementWidth } from '@site/src/components/viz/hooks'
import { dashClass, type Dash, Legend, Marker, seriesColor } from '@site/src/components/viz/marks'
import * as d3 from 'd3'
import React, { type ReactNode, useState } from 'react'

const SMALL_HEIGHT = 220
const LARGE_HEIGHT = 320
const SEARCH_HEIGHT = 240
const LOG_TICKS = [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]
/** The ticks of a logarithmic density axis. */
const DENSITY_TICKS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
/** The populations of the row of examples: the youngest, one of middle age and the oldest. */
const EXAMPLES = [0, 4, 9]
const FIT = seriesColor(0)

/** The predictions drawn over the stars, with their lines. */
type Predicted = 'tracer' | 'jeansSigma' | 'jeans' | 'sheet'

const PREDICTIONS: readonly (readonly [Predicted, string, Dash, string])[] = [
  ['tracer', 'tracer', 'solid', seriesColor(1)],
  ['jeansSigma', 'Jeans, measured $\\sigma_z(z)$', 'dashdot', seriesColor(2)],
  ['jeans', 'Jeans, $\\sigma_z(z)$ and tilt', 'dotted', seriesColor(3)],
  ['sheet', 'sheet', 'dashed', seriesColor(4)]
]

/** The mass per unit height of the windowed exponential fit, normalized to the stars inside the window. */
function fitted(bin: PopulationFit, z: readonly number[]): readonly number[] {
  const { hz, window } = bin
  if (!Number.isFinite(hz) || !Number.isFinite(window)) {
    return z.map(() => Number.NaN)
  }
  let mass = 0
  bin.density.forEach((d, i) => {
    const lo = bin.edges[i]
    const hi = Math.min(bin.edges[i + 1], window)
    if (hi > lo) {
      mass += d * (hi - lo)
    }
  })
  const norm = mass / (hz * -Math.expm1(-window / hz))
  return z.map((height) => (height <= window ? norm * Math.exp(-height / hz) : Number.NaN))
}

interface ProfileProps {
  readonly bin: PopulationFit
  readonly width: number
  readonly height: number
  readonly predictions: boolean
}

/** The heights of the stars of one population, its windowed fit and, if asked, the three predictions. */
function Profile({ bin, width, height, predictions }: ProfileProps): ReactNode {
  const steps: (readonly [number, number])[] = []
  bin.density.forEach((d, i) => {
    if (d > 0) {
      steps.push([bin.edges[i], d], [bin.edges[i + 1], d])
    }
  })
  const values = [
    ...bin.density.filter((d) => d > 0),
    ...(predictions ? PREDICTIONS.flatMap(([key]) => bin[key]) : [])
  ].filter((v) => Number.isFinite(v) && v > 0)
  const top = (d3.max(values) ?? 1) * 1.5
  const bottom = Math.max(
    top * 1e-3,
    (d3.min(bin.density.filter((d) => d > 0)) ?? top * 1e-3) / 1.5
  )
  const zMax = bin.edges[bin.edges.length - 1]
  const fit = fitted(bin, bin.z)
  return (
    <Chart
      width={width}
      height={height}
      xLabel="$|z|$ (kpc)"
      yLabel="mass per unit height (kpc$^{-1}$)"
    >
      {(inner) => {
        const x = d3.scaleLinear().domain([0, zMax]).range([0, inner.width])
        const y = d3.scaleLog().domain([bottom, top]).range([inner.height, 0])
        const curve = (ys: readonly number[]) =>
          linePath(
            bin.z
              .map((z, i) => [z, ys[i]] as const)
              .filter(([, v]) => Number.isFinite(v) && v >= bottom)
              .map(([z, v]) => [x(z), y(v)] as const)
          )
        return (
          <>
            <Axis
              scale={y}
              orientation="left"
              tickValues={ticksWithin(DENSITY_TICKS, [bottom, top])}
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
            <g clipPath={inner.clip}>
              {Number.isFinite(bin.window) ? (
                <>
                  <rect
                    x={0}
                    y={0}
                    width={x(bin.window)}
                    height={inner.height}
                    className="viz-window"
                  />
                  <rect
                    x={0}
                    y={0}
                    width={x(bin.window / 2)}
                    height={inner.height}
                    className="viz-window viz-window--half"
                  />
                </>
              ) : null}
              <path
                d={linePath(steps.map(([z, d]) => [x(z), y(d)] as const))}
                className="viz-step"
              />
              {predictions
                ? PREDICTIONS.map(([key, label, dash, color]) => (
                    <path
                      key={label}
                      d={curve(bin[key])}
                      className={dashClass(dash)}
                      style={{ stroke: color }}
                    />
                  ))
                : null}
              <path
                d={curve(fit)}
                className={dashClass('solid')}
                style={{ stroke: FIT, strokeWidth: 2.5 }}
              />
            </g>
          </>
        )
      }}
    </Chart>
  )
}

/** One example of the row, as wide as its cell of the grid. */
function Example({ bin }: { readonly bin: PopulationFit }): ReactNode {
  const [ref, width] = useElementWidth<HTMLDivElement>()
  return (
    <div ref={ref}>
      <p className="viz-panel-title">
        <MathLabel
          label={`${ageLabel(bin)}: $\\sigma_z = ${bin.sigmaZ.toFixed(1)}$ km/s, $h_Z = ${bin.hz.toFixed(2)}$ kpc`}
        />
      </p>
      <Profile bin={bin} width={width} height={SMALL_HEIGHT} predictions={false} />
    </div>
  )
}

/** The windowed h_Z at every trial window of the search, with the line L = 2h. */
function Search({
  bin,
  width
}: {
  readonly bin: PopulationFit
  readonly width: number
}): ReactNode {
  const points = bin.windows.map((l, i) => [l, bin.heights[i]] as const)
  const finite = points.filter(([, h]) => Number.isFinite(h))
  const hs = finite.map(([, h]) => h)
  const yDomain: readonly [number, number] = [
    Math.max((d3.min(hs) ?? 0.02) / 1.5, 0.01),
    Math.min((d3.max(hs) ?? 10) * 1.5, 100)
  ]
  return (
    <Chart width={width} height={SEARCH_HEIGHT} xLabel="window $L$ (kpc)" yLabel="$h(L)$ (kpc)">
      {(inner) => {
        const x = d3.scaleLog().domain([0.05, 12]).range([0, inner.width])
        const y = d3.scaleLog().domain(yDomain).range([inner.height, 0])
        return (
          <>
            <Axis
              scale={y}
              orientation="left"
              tickValues={ticksWithin(LOG_TICKS, yDomain)}
              tickFormat={(v) => String(v)}
              gridLength={inner.width}
            />
            <Axis
              scale={x}
              orientation="bottom"
              transform={`translate(0,${inner.height})`}
              tickValues={LOG_TICKS}
              tickFormat={(v) => String(v)}
              gridLength={inner.height}
            />
            <g clipPath={inner.clip}>
              <path
                d={linePath([0.05, 12].map((l) => [x(l), y(l / 2)] as const))}
                className="viz-guide viz-line--dashed"
              />
              {finite.map(([l, h]) => (
                <Marker key={l} shape="circle" x={x(l)} y={y(h)} open />
              ))}
              {Number.isFinite(bin.hz) ? (
                <Marker shape="diamond" x={x(bin.window)} y={y(bin.hz)} color={seriesColor(2)} />
              ) : null}
            </g>
          </>
        )
      }}
    </Chart>
  )
}

function ageLabel(bin: PopulationFit): string {
  return `${bin.ageLo}-${bin.ageHi} Gyr`
}

/**
 * Examples of the fit of the scale height in one halo: the heights of the star particles of a
 * population, weighted by initial mass, with the window of the fit and the exponential that it
 * fits, and, for any population, the three predictions and the search for the window.
 */
export default function FitChart(): ReactNode {
  const [ref, width] = useElementWidth<HTMLDivElement>()
  const [label, setLabel] = useState(DEFAULT_HALO)
  const [chosen, setChosen] = useState(9)
  const { value, error } = useConcepts(haloOf(label))
  const bin = value?.bins[chosen]
  return (
    <div className="viz" ref={ref}>
      <div className="viz-controls" style={{ borderTop: 'none', marginTop: 0, paddingTop: 0 }}>
        <HaloChips selected={label} onSelect={setLabel} />
      </div>
      {value === null || bin === undefined ? (
        <Pending error={error} />
      ) : (
        <>
          <div className="viz-row">
            {EXAMPLES.map((i) => (
              <Example key={i} bin={value.bins[i]} />
            ))}
          </div>
          <Legend
            entries={[
              {
                label: 'star particles, weighted by initial mass',
                dash: 'solid',
                color: 'var(--viz-muted)'
              },
              { label: 'the windowed exponential, $h_Z$', dash: 'solid', color: FIT },
              { label: 'the windows $L = 2h_Z$ and $L/2$', band: true }
            ]}
          />
          <p className="viz-panel-title">
            <MathLabel
              label={`${label}, ${ageLabel(bin)}: ${bin.nStars.toLocaleString('en-US')} star particles, $\\sigma_z = ${bin.sigmaZ.toFixed(1)}$ km/s, $h_Z = ${bin.hz.toFixed(3)}$ kpc in $L = ${bin.window.toFixed(2)}$ kpc`}
            />
          </p>
          <Profile bin={bin} width={width} height={LARGE_HEIGHT} predictions />
          <Legend
            entries={[
              { label: 'star particles', dash: 'solid', color: 'var(--viz-muted)' },
              { label: 'windowed exponential', dash: 'solid', color: FIT },
              ...PREDICTIONS.map(([, predictionLabel, dash, color]) => ({
                label: predictionLabel,
                dash,
                color
              }))
            ]}
          />
          <Search bin={bin} width={width} />
          <Legend
            entries={[
              { label: '$h(L)$ at the trial windows of the search', marker: 'circle', open: true },
              { label: '$L = 2h$', dash: 'dashed' },
              { label: 'the self-consistent window', marker: 'diamond', color: seriesColor(2) }
            ]}
          />
          <div className="viz-controls">
            <Chips
              label="Population"
              options={value.bins.map((b, i) => [String(i), ageLabel(b)] as const)}
              isPressed={(option) => Number(option) === chosen}
              onToggle={(option) => setChosen(Number(option))}
            />
          </div>
        </>
      )}
    </div>
  )
}
