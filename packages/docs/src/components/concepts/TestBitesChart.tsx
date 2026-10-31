import {
  dfMoments,
  geomspace,
  harmonicPotential,
  matchedDf,
  populationMoments,
  scaledProfileHz,
  SOLAR_TOY,
  toyPotential,
  trapezoid,
  Z_GRID
} from '@site/src/components/concepts/physics'
import MathLabel from '@site/src/components/MathLabel'
import { Chips } from '@site/src/components/status/controls'
import Chart, { Axis, bandPath, linePath } from '@site/src/components/viz/Chart'
import { useElementWidth } from '@site/src/components/viz/hooks'
import { dashClass, type Dash, Legend, seriesColor } from '@site/src/components/viz/marks'
import Slider from '@site/src/components/viz/Slider'
import * as d3 from 'd3'
import React, { type ReactNode, useEffect, useMemo, useState } from 'react'

const POTENTIAL_HEIGHT = 240
const PROFILE_HEIGHT = 240
const RATIO_HEIGHT = 260
const SIGMA_RANGE: readonly [number, number] = [6, 60]
const SIGMA_TICKS = [6, 10, 20, 40, 60]
/** Where a population lives: Delta Phi up to this many sigma_z^2 holds all but e^-2 of an isothermal one. */
const LIVED_IN = 2
/** The harmonic core: where the potential is within 10 percent of the harmonic one, in |ln|. */
const CORE_TOLERANCE = 0.1
const N_HEIGHTS = 180
/** The heights of a population reach where Delta Phi is this many sigma_z^2. */
const REACH = 150

type PotentialMode = 'toy' | 'harmonic'

const POTENTIALS: Record<PotentialMode, (z: number) => number> = {
  toy: (z) => toyPotential(SOLAR_TOY, z),
  harmonic: (z) => harmonicPotential(SOLAR_TOY, z)
}

/** The three shapes of the distribution function exp[-(E/E0)^q], with the line of each. */
const SHAPES: readonly (readonly [string, number, Dash, string])[] = [
  ['isothermal, $q = 1$', 1, 'solid', seriesColor(0)],
  ['heavier tails, $q = 0.7$', 0.7, 'dashed', seriesColor(1)],
  ['sharper cutoff, $q = 1.5$', 1.5, 'dashdot', seriesColor(2)]
]

/** The heights (kpc) on which a population of dispersion `sigma` is integrated, denser near the plane. */
function heights(phi: (z: number) => number, sigma: number): readonly number[] {
  let lo = 0
  let hi = 300
  for (let i = 0; i < 50; i += 1) {
    const mid = 0.5 * (lo + hi)
    if (phi(mid) < REACH * sigma ** 2) {
      lo = mid
    } else {
      hi = mid
    }
  }
  return Array.from({ length: N_HEIGHTS }, (_, i) => hi * (i / (N_HEIGHTS - 1)) ** 2)
}

/**
 * The density (unit mass), the rms height and the windowed h_Z of each shape, all with a
 * population-averaged v_z^2 of sigma^2.
 */
interface Populations {
  readonly z: readonly number[]
  readonly rho: readonly (readonly number[])[]
  readonly rms: readonly number[]
  readonly windowed: readonly number[]
}

/** The windowed h_Z of a profile `rho` on the heights `z`, interpolated onto Z_GRID. */
function windowedHeight(z: readonly number[], rho: readonly number[]): number {
  const at = d3.scaleLinear(z, rho).clamp(true)
  const top = z[z.length - 1]
  return scaledProfileHz(Z_GRID.map((h) => (h > top ? 0 : at(h)))).hz
}

function populations(mode: PotentialMode, sigma: number): Populations {
  const phi = POTENTIALS[mode]
  const z = heights(phi, sigma)
  const phiZ = z.map(phi)
  const out = SHAPES.map(([, q]) => {
    const df = q === 1 ? { e0: sigma ** 2, q: 1 } : matchedDf(q, sigma, phiZ, z)
    const { rho, v2 } = dfMoments(df, phiZ)
    const mass = trapezoid(rho, z)
    return {
      rho: rho.map((r) => r / mass),
      rms: populationMoments(rho, v2, z).rmsZ,
      windowed: windowedHeight(z, rho)
    }
  })
  return {
    z,
    rho: out.map((o) => o.rho),
    rms: out.map((o) => o.rms),
    windowed: out.map((o) => o.windowed)
  }
}

/** How the heights of the shapes compare: the rms height or the windowed h_Z. */
type Measure = 'rms' | 'windowed'

/** The height of each non-isothermal shape over that of the isothermal one, at one dispersion. */
interface RatioPoint {
  readonly sigma: number
  readonly toy: Readonly<Record<Measure, readonly number[]>>
  readonly harmonic: Readonly<Record<Measure, readonly number[]>>
}

const RATIO_SIGMAS = geomspace(SIGMA_RANGE[0], SIGMA_RANGE[1], 12)

/** The ratios over RATIO_SIGMAS, computed one dispersion per frame so that the page stays responsive. */
function useRatios(): readonly RatioPoint[] {
  const [points, setPoints] = useState<readonly RatioPoint[]>([])
  useEffect(() => {
    let cancelled = false
    let next = 0
    const step = (): void => {
      if (cancelled || next >= RATIO_SIGMAS.length) {
        return
      }
      const sigma = RATIO_SIGMAS[next]
      const ratios = (mode: PotentialMode) => {
        const { rms, windowed } = populations(mode, sigma)
        return {
          rms: rms.slice(1).map((r) => r / rms[0]),
          windowed: windowed.slice(1).map((h) => h / windowed[0])
        }
      }
      const point = { sigma, toy: ratios('toy'), harmonic: ratios('harmonic') }
      setPoints((current) => [...current, point])
      next += 1
      window.requestAnimationFrame(step)
    }
    window.requestAnimationFrame(step)
    return () => {
      cancelled = true
    }
  }, [])
  return points
}

interface PotentialProps {
  readonly sigma: number
  readonly zMax: number
  readonly width: number
}

/** Delta Phi / sigma_z^2 of the toy disk and of its harmonic core, with where the population lives. */
function PotentialPanel({ sigma, zMax, width }: PotentialProps): ReactNode {
  const z = d3.range(0, 241).map((i) => (zMax * i) / 240)
  const toy = z.map((h) => POTENTIALS.toy(h) / sigma ** 2)
  const harmonic = z.map((h) => POTENTIALS.harmonic(h) / sigma ** 2)
  const coreEnd =
    z.find((h, i) => i > 0 && Math.abs(Math.log(toy[i] / harmonic[i])) > CORE_TOLERANCE) ?? zMax
  const yMax = 6
  return (
    <Chart
      width={width}
      height={POTENTIAL_HEIGHT}
      xLabel="$|z|$ (kpc)"
      yLabel="$\Delta\Phi/\sigma_z^2$"
    >
      {(inner) => {
        const x = d3.scaleLinear().domain([0, zMax]).range([0, inner.width])
        const y = d3.scaleLinear().domain([0, yMax]).range([inner.height, 0])
        return (
          <>
            <Axis scale={y} orientation="left" ticks={4} gridLength={inner.width} />
            <Axis
              scale={x}
              orientation="bottom"
              transform={`translate(0,${inner.height})`}
              ticks={6}
              gridLength={inner.height}
            />
            <g clipPath={inner.clip}>
              <rect x={0} y={0} width={x(coreEnd)} height={inner.height} className="viz-window" />
              <rect
                x={0}
                y={y(LIVED_IN)}
                width={inner.width}
                height={inner.height - y(LIVED_IN)}
                className="viz-window viz-window--half"
              />
              <path
                d={linePath(harmonic.map((v, i) => [x(z[i]), y(v)] as const))}
                className={dashClass('dashed')}
              />
              <path
                d={linePath(toy.map((v, i) => [x(z[i]), y(v)] as const))}
                className={dashClass('solid')}
              />
              <text
                x={Math.min(x(coreEnd), inner.width) - 4}
                y={14}
                className="viz-text"
                textAnchor="end"
              >
                harmonic core
              </text>
              <text x={inner.width - 4} y={y(LIVED_IN) - 4} className="viz-text" textAnchor="end">
                where most of its stars live
              </text>
            </g>
          </>
        )
      }}
    </Chart>
  )
}

interface ProfilesProps {
  readonly pops: Populations
  readonly zMax: number
  readonly width: number
}

/** The profiles of the three shapes, each of unit mass. */
function ProfilesPanel({ pops, zMax, width }: ProfilesProps): ReactNode {
  const shown = pops.z.findIndex((h) => h > zMax)
  const last = shown < 0 ? pops.z.length : shown + 1
  const top = (d3.max(pops.rho.flatMap((rho) => rho.slice(0, last))) ?? 1) * 1.08
  return (
    <Chart
      width={width}
      height={PROFILE_HEIGHT}
      xLabel="$|z|$ (kpc)"
      yLabel="density per unit mass (kpc$^{-1}$)"
    >
      {(inner) => {
        const x = d3.scaleLinear().domain([0, zMax]).range([0, inner.width])
        const y = d3.scaleLinear().domain([0, top]).range([inner.height, 0])
        return (
          <>
            <Axis scale={y} orientation="left" ticks={4} gridLength={inner.width} />
            <Axis
              scale={x}
              orientation="bottom"
              transform={`translate(0,${inner.height})`}
              ticks={6}
              gridLength={inner.height}
            />
            <g clipPath={inner.clip}>
              {SHAPES.map(([label, , dash, color], s) => (
                <path
                  key={label}
                  d={linePath(
                    pops.rho[s].slice(0, last).map((v, i) => [x(pops.z[i]), y(v)] as const)
                  )}
                  className={dashClass(dash)}
                  style={{ stroke: color }}
                />
              ))}
            </g>
          </>
        )
      }}
    </Chart>
  )
}

/** The rms height of each shape over the isothermal one against sigma_z, in both potentials. */
function RatioPanel({
  points,
  measure,
  sigma,
  width
}: {
  readonly points: readonly RatioPoint[]
  readonly measure: Measure
  readonly sigma: number
  readonly width: number
}): ReactNode {
  const values = points.flatMap((p) => [...p.toy[measure], ...p.harmonic[measure]])
  // At least 15 percent either way, so that the band of 10 percent reads as a band.
  const spread = Math.max(0.15, ...values.map((v) => Math.abs(Math.log(v))).map((v) => v * 1.15))
  const yDomain: readonly [number, number] = [Math.exp(-spread), Math.exp(spread)]
  return (
    <Chart
      width={width}
      height={RATIO_HEIGHT}
      xLabel="$\sigma_z$ (km/s)"
      yLabel={
        measure === 'rms' ? '$z_\\mathrm{rms}/z_\\mathrm{rms,iso}$' : '$h_Z/h_{Z,\\mathrm{iso}}$'
      }
    >
      {(inner) => {
        const x = d3.scaleLog().domain(SIGMA_RANGE).range([0, inner.width])
        const y = d3.scaleLog().domain(yDomain).range([inner.height, 0])
        const band = (tolerance: number) =>
          bandPath(
            SIGMA_RANGE.map((s) => [x(s), y(Math.exp(-tolerance))] as const),
            SIGMA_RANGE.map((s) => [x(s), y(Math.exp(tolerance))] as const)
          )
        return (
          <>
            <Axis
              scale={y}
              orientation="left"
              ticks={5}
              tickFormat={(v) => v.toFixed(2)}
              gridLength={inner.width}
            />
            <Axis
              scale={x}
              orientation="bottom"
              transform={`translate(0,${inner.height})`}
              tickValues={SIGMA_TICKS}
              tickFormat={(v) => String(v)}
              gridLength={inner.height}
            />
            <g clipPath={inner.clip}>
              <path d={band(0.1)} className="viz-band" />
              <line
                x1={0}
                x2={inner.width}
                y1={y(1)}
                y2={y(1)}
                className="viz-line viz-line--thin"
              />
              {SHAPES.slice(1).map(([label, , , color], s) => (
                <g key={label}>
                  <path
                    d={linePath(
                      points.map((p) => [x(p.sigma), y(p.harmonic[measure][s])] as const)
                    )}
                    className={dashClass('dotted')}
                    style={{ stroke: color }}
                  />
                  <path
                    d={linePath(points.map((p) => [x(p.sigma), y(p.toy[measure][s])] as const))}
                    className={dashClass('solid')}
                    style={{ stroke: color }}
                  />
                </g>
              ))}
              <line x1={x(sigma)} x2={x(sigma)} y1={0} y2={inner.height} className="viz-guide" />
            </g>
          </>
        )
      }}
    </Chart>
  )
}

/**
 * Why agreement of the tracer with cold populations is guaranteed and with hot ones is a test.
 * Three populations whose distribution functions of the vertical energy have different shapes,
 * all with the same population-averaged v_z^2, are put in the toy disk or in the harmonic
 * potential of its midplane density. Their rms heights agree in the harmonic potential, as the
 * virial theorem demands, and in the toy disk only while they stay in its harmonic core.
 */
export default function TestBitesChart(): ReactNode {
  const [ref, width] = useElementWidth<HTMLDivElement>()
  const [sigma, setSigma] = useState(15)
  const [mode, setMode] = useState<PotentialMode>('toy')
  const [measure, setMeasure] = useState<Measure>('rms')
  const pops = useMemo(() => populations(mode, sigma), [mode, sigma])
  const ratios = useRatios()
  const zMax = Math.min(Math.max(4 * pops.rms[0], 0.4), 12)
  return (
    <div className="viz" ref={ref}>
      <PotentialPanel sigma={sigma} zMax={zMax} width={width} />
      <Legend
        entries={[
          { label: 'toy disk', dash: 'solid' },
          { label: 'harmonic, with its midplane density', dash: 'dashed' },
          { label: 'its harmonic core, within 10 percent', band: true }
        ]}
      />
      <ProfilesPanel pops={pops} zMax={zMax} width={width} />
      <Legend entries={SHAPES.map(([label, , dash, color]) => ({ label, dash, color }))} />
      <RatioPanel points={ratios} measure={measure} sigma={sigma} width={width} />
      <Legend
        entries={[
          { label: 'in the toy disk', dash: 'solid' },
          { label: 'in the harmonic potential', dash: 'dotted' },
          { label: 'within 10 percent', band: true }
        ]}
      />
      <div className="viz-controls">
        <Slider
          label="$\sigma_z$"
          value={sigma}
          min={SIGMA_RANGE[0]}
          max={SIGMA_RANGE[1]}
          log
          format={(v) => `$${v.toFixed(0)}$ km/s`}
          onChange={setSigma}
        />
        <Chips
          label="Potential of the profiles"
          options={[
            ['toy', 'toy disk'],
            ['harmonic', 'harmonic']
          ]}
          isPressed={(option) => option === mode}
          onToggle={setMode}
        />
        <Chips
          label="Height compared"
          options={[
            ['rms', 'rms height'],
            ['windowed', 'windowed $h_Z$']
          ]}
          isPressed={(option) => option === measure}
          onToggle={setMeasure}
        />
      </div>
      <p className="viz-note">
        <MathLabel
          label={`At $\\sigma_z = ${sigma.toFixed(0)}$ km/s in the ${mode === 'toy' ? 'toy disk' : 'harmonic potential'}, the rms heights are ${pops.rms.map((r) => `$${r.toFixed(3)}$`).join(', ')} kpc and the windowed $h_Z$ ${pops.windowed.map((h) => `$${h.toFixed(3)}$`).join(', ')} kpc, for the isothermal population, the heavier tails and the sharper cutoff.`}
        />
      </p>
    </div>
  )
}
