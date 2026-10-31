import {
  geomspace,
  profileHz,
  profileMean,
  scaledProfileHz,
  sech2,
  SOLAR_TOY,
  toyPotential,
  tracerProfile,
  trapezoid,
  truncatedExpScale,
  Z_GRID
} from '@site/src/components/concepts/physics'
import MathLabel from '@site/src/components/MathLabel'
import { Chips } from '@site/src/components/status/controls'
import Chart, { Axis, linePath, ticksWithin } from '@site/src/components/viz/Chart'
import { useElementWidth } from '@site/src/components/viz/hooks'
import { dashClass, Legend, Marker, seriesColor } from '@site/src/components/viz/marks'
import Slider from '@site/src/components/viz/Slider'
import * as d3 from 'd3'
import React, { type ReactNode, useMemo, useState } from 'react'

const PROFILE_HEIGHT = 280
const SEARCH_HEIGHT = 280
const WINDOW_RANGE: readonly [number, number] = [0.1, 12]
const LOG_TICKS = [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]
/** The fixed window of the example, Z_lim = 1 kpc. */
const FIXED_WINDOW = 1
const SEARCH_WINDOWS = geomspace(0.05, 12, 140)
const FIT = seriesColor(0)
const HALF = seriesColor(1)

type Family = 'exponential' | 'sech2' | 'mixture' | 'tracer'

/** The profiles of the example, each of unit mass over 0-12 kpc. */
const FAMILIES: Record<Family, readonly [string, () => readonly number[]]> = {
  exponential: ['exponential, $h = 0.4$ kpc', () => Z_GRID.map((z) => Math.exp(-z / 0.4))],
  sech2: ['$\\mathrm{sech}^2$, $z_0 = 0.8$ kpc', () => Z_GRID.map((z) => sech2(z, 0.8))],
  mixture: [
    'two exponentials, 0.25 and 1 kpc',
    () => Z_GRID.map((z) => (0.6 * Math.exp(-z / 0.25)) / 0.25 + (0.4 * Math.exp(-z)) / 1)
  ],
  tracer: [
    'tracer of $35$ km/s in the toy disk',
    () => tracerProfile((z) => toyPotential(SOLAR_TOY, z), 35)
  ]
}

function unitMass(rho: readonly number[]): readonly number[] {
  const mass = trapezoid(rho, Z_GRID)
  return rho.map((r) => r / mass)
}

/** The mass of the profile `rho` on Z_GRID below `limit`. */
function massBelow(rho: readonly number[], limit: number): number {
  const inside = Z_GRID.findIndex((z) => z > limit)
  const end = inside < 0 ? Z_GRID.length : inside
  const z = [...Z_GRID.slice(0, end), limit]
  const values = [...rho.slice(0, end), d3.scaleLinear(Z_GRID, rho)(limit)]
  return trapezoid(values, z)
}

/** The truncated exponential of scale `h` on [0, `limit`] that holds `mass`, at the heights `z`. */
function truncatedExponential(
  z: readonly number[],
  h: number,
  limit: number,
  mass: number
): readonly number[] {
  const norm = mass / (h * -Math.expm1(-limit / h))
  return z.map((height) => (height <= limit ? norm * Math.exp(-height / h) : Number.NaN))
}

interface Fit {
  readonly window: number
  readonly mean: number
  readonly scale: number
  readonly mass: number
}

function fitInside(rho: readonly number[], window: number): Fit {
  const mean = profileMean(rho, Z_GRID, window)
  return { window, mean, scale: truncatedExpScale(mean, window), mass: massBelow(rho, window) }
}

interface ProfileProps {
  readonly rho: readonly number[]
  readonly fits: readonly [Fit, Fit]
  readonly zMax: number
  readonly logScale: boolean
  readonly width: number
}

/** The profile, the window and its half, and the exponential fitted inside each. */
function ProfilePanel({ rho, fits, zMax, logScale, width }: ProfileProps): ReactNode {
  const end = Z_GRID.findIndex((z) => z > zMax)
  const z = Z_GRID.slice(0, end < 0 ? Z_GRID.length : end + 1)
  const shown = rho.slice(0, z.length)
  const top = (d3.max(shown) ?? 1) * 1.15
  const bottom = Math.max(top * 1e-3, d3.min(shown.filter((v) => v > 0)) ?? top * 1e-3)
  return (
    <Chart
      width={width}
      height={PROFILE_HEIGHT}
      xLabel="$|z|$ (kpc)"
      yLabel="density per unit mass (kpc$^{-1}$)"
    >
      {(inner) => {
        const x = d3.scaleLinear().domain([0, zMax]).range([0, inner.width])
        const y = logScale
          ? d3.scaleLog().domain([bottom, top]).range([inner.height, 0])
          : d3.scaleLinear().domain([0, top]).range([inner.height, 0])
        const curve = (values: readonly number[]) =>
          linePath(
            values
              .map((v, i) => [z[i], v] as const)
              .filter(([, v]) => Number.isFinite(v) && (!logScale || v >= bottom))
              .map(([h, v]) => [x(h), y(v)] as const)
          )
        return (
          <>
            <Axis
              scale={y}
              orientation="left"
              ticks={4}
              tickFormat={(v) => (logScale ? d3.format('.1~g')(v) : d3.format('.2~f')(v))}
              gridLength={inner.width}
            />
            <Axis
              scale={x}
              orientation="bottom"
              transform={`translate(0,${inner.height})`}
              ticks={6}
              gridLength={inner.height}
            />
            <g clipPath={inner.clip}>
              <rect
                x={0}
                y={0}
                width={x(fits[0].window)}
                height={inner.height}
                className="viz-window"
              />
              <rect
                x={0}
                y={0}
                width={x(fits[1].window)}
                height={inner.height}
                className="viz-window viz-window--half"
              />
              <path d={curve(shown)} className={dashClass('solid')} />
              {fits.map((fit, i) =>
                Number.isFinite(fit.scale) ? (
                  <path
                    key={fit.window}
                    d={curve(truncatedExponential(z, fit.scale, fit.window, fit.mass))}
                    className={dashClass(i === 0 ? 'dashed' : 'dotted')}
                    style={{ stroke: i === 0 ? FIT : HALF }}
                  />
                ) : null
              )}
              {fits.map((fit, i) => (
                <g key={`mean-${fit.window}`}>
                  <line
                    x1={x(fit.mean)}
                    x2={x(fit.mean)}
                    y1={inner.height}
                    y2={inner.height - 14}
                    className="viz-line"
                    style={{ stroke: i === 0 ? FIT : HALF }}
                  />
                  <line
                    x1={x(fit.window / 2)}
                    x2={x(fit.window / 2)}
                    y1={inner.height}
                    y2={inner.height - 8}
                    className="viz-line viz-line--thin"
                    style={{ stroke: i === 0 ? FIT : HALF }}
                  />
                </g>
              ))}
            </g>
          </>
        )
      }}
    </Chart>
  )
}

interface SearchProps {
  readonly curve: readonly (readonly [number, number])[]
  readonly window: number
  readonly current: number
  readonly selfConsistent: { readonly hz: number; readonly window: number }
  readonly fixed: number
  readonly width: number
}

/** h(L) against L, with the line L = 2h whose first crossing is the self-consistent window. */
function SearchPanel({
  curve,
  window,
  current,
  selfConsistent,
  fixed,
  width
}: SearchProps): ReactNode {
  const finite = curve.filter(([, h]) => Number.isFinite(h))
  const hs = finite.map(([, h]) => h)
  const yDomain: readonly [number, number] = [
    Math.max((d3.min(hs) ?? 0.02) / 1.5, 0.01),
    Math.min((d3.max(hs) ?? 10) * 1.5, 100)
  ]
  // The runs of finite values, so that the gaps where the fit has no root stay gaps.
  const runs: (readonly [number, number])[][] = [[]]
  for (const point of curve) {
    if (Number.isFinite(point[1])) {
      runs[runs.length - 1].push(point)
    } else if (runs[runs.length - 1].length > 0) {
      runs.push([])
    }
  }
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
              <text x={x(0.12)} y={y(0.06) - 6} className="viz-text">
                L = 2h
              </text>
              <line x1={x(fixed)} x2={x(fixed)} y1={0} y2={inner.height} className="viz-guide" />
              <text x={x(fixed) + 4} y={14} className="viz-text">
                a fixed window, 1 kpc
              </text>
              {runs
                .filter((run) => run.length > 1)
                .map((run) => (
                  <path
                    key={run[0][0]}
                    d={linePath(run.map(([l, h]) => [x(l), y(h)] as const))}
                    className={dashClass('solid')}
                  />
                ))}
              {Number.isFinite(selfConsistent.hz) ? (
                <Marker
                  shape="diamond"
                  x={x(selfConsistent.window)}
                  y={y(selfConsistent.hz)}
                  color={seriesColor(2)}
                />
              ) : null}
              {Number.isFinite(current) ? (
                <Marker shape="circle" x={x(window)} y={y(current)} open color={FIT} />
              ) : null}
            </g>
          </>
        )
      }}
    </Chart>
  )
}

/**
 * The windowed scale height of `experiments.heights` on four profiles: the exponential fitted
 * inside a window L and inside L/2, their average h(L), and the self-consistent window, the
 * smallest L with L = 2 h(L).
 */
export default function WindowChart(): ReactNode {
  const [ref, width] = useElementWidth<HTMLDivElement>()
  const [family, setFamily] = useState<Family>('sech2')
  const [logScale, setLogScale] = useState(false)
  const rho = useMemo(() => unitMass(FAMILIES[family][1]()), [family])
  const selfConsistent = useMemo(() => scaledProfileHz(rho), [rho])
  const curve = useMemo(
    () => SEARCH_WINDOWS.map((l) => [l, profileHz(rho, Z_GRID, l)] as const),
    [rho]
  )
  const [chosen, setChosen] = useState<number | null>(null)
  const shownWindow = chosen ?? (Number.isFinite(selfConsistent.window) ? selfConsistent.window : 1)
  const fits: readonly [Fit, Fit] = [fitInside(rho, shownWindow), fitInside(rho, shownWindow / 2)]
  const current = profileHz(rho, Z_GRID, shownWindow)
  const fixed = profileHz(rho, Z_GRID, FIXED_WINDOW)
  const zMax = Math.min(12, Math.max(1.4 * shownWindow, 3 * selfConsistent.window || 1, 1))
  const scale = (v: number) => (Number.isFinite(v) ? `$${v.toFixed(3)}$ kpc` : 'no root')
  return (
    <div className="viz" ref={ref}>
      <ProfilePanel rho={rho} fits={fits} zMax={zMax} logScale={logScale} width={width} />
      <Legend
        entries={[
          { label: FAMILIES[family][0], dash: 'solid' },
          { label: 'fit inside $L$', dash: 'dashed', color: FIT },
          { label: 'fit inside $L/2$', dash: 'dotted', color: HALF },
          { label: 'the windows $L$ and $L/2$', band: true }
        ]}
      />
      <SearchPanel
        curve={curve}
        window={shownWindow}
        current={current}
        selfConsistent={selfConsistent}
        fixed={FIXED_WINDOW}
        width={width}
      />
      <Legend
        entries={[
          { label: '$h(L)$, the mean of the two fits', dash: 'solid' },
          { label: 'the window $L$ of the slider', marker: 'circle', open: true, color: FIT },
          { label: 'the self-consistent window', marker: 'diamond', color: seriesColor(2) }
        ]}
      />
      <div className="viz-controls">
        <Chips
          label="Profile"
          options={(Object.keys(FAMILIES) as Family[]).map(
            (name) => [name, FAMILIES[name][0]] as const
          )}
          isPressed={(option) => option === family}
          onToggle={(option) => {
            setFamily(option)
            setChosen(null)
          }}
        />
        <Slider
          label="$L$"
          value={shownWindow}
          min={WINDOW_RANGE[0]}
          max={WINDOW_RANGE[1]}
          log
          format={(v) => `$${v.toFixed(2)}$ kpc`}
          onChange={setChosen}
        />
        <Chips
          label="Density axis"
          options={[
            ['linear', 'linear'],
            ['log', 'logarithmic']
          ]}
          isPressed={(option) => (option === 'log') === logScale}
          onToggle={(option) => setLogScale(option === 'log')}
        />
      </div>
      <p className="viz-note">
        <MathLabel
          label={`Inside $L = ${shownWindow.toFixed(2)}$ kpc the mean height is $${fits[0].mean.toFixed(3)}$ kpc against $L/2 = ${(shownWindow / 2).toFixed(3)}$ kpc, and the fit gives ${scale(fits[0].scale)}. Inside $L/2$ the fit gives ${scale(fits[1].scale)}, so $h(L)$ is ${scale(current)}. The self-consistent window is $L = ${selfConsistent.window.toFixed(2)}$ kpc with $h_Z = ${selfConsistent.hz.toFixed(3)}$ kpc, whereas a fixed window of 1 kpc gives ${scale(fixed)}.`}
        />
      </p>
    </div>
  )
}
