import {
  diskScale,
  geomspace,
  linspace,
  scaledProfileHz,
  sech2,
  sheetScale,
  SOLAR_TOY,
  surfaceDensityWithin,
  toyPotential,
  type ToyDisk,
  tracerProfile,
  trapezoid,
  Z_GRID
} from '@site/src/components/concepts/physics'
import MathLabel from '@site/src/components/MathLabel'
import { Chips } from '@site/src/components/status/controls'
import Chart, { Axis, bandPath, linePath, ticksWithin } from '@site/src/components/viz/Chart'
import { useElementWidth } from '@site/src/components/viz/hooks'
import { dashClass, Legend, Marker, seriesColor } from '@site/src/components/viz/marks'
import Slider from '@site/src/components/viz/Slider'
import * as d3 from 'd3'
import React, { type ReactNode, useMemo, useState } from 'react'

const PROFILE_HEIGHT = 280
const SCALING_HEIGHT = 300
const SIGMA_RANGE: readonly [number, number] = [5, 80]
const SIGMA_TICKS = [5, 10, 20, 40, 80]
const HEIGHT_TICKS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]
/** The layer whose mass per area is the Sigma of the sheet, as in the paper (kpc). */
const SHEET_LIMIT_KPC = 1.1
const PROFILE_POINTS = 300
const TRACER = seriesColor(0)
const SHEET = seriesColor(1)

type HaloMode = 'with' | 'without'

interface Model {
  readonly disk: ToyDisk
  /** The Sigma (Msun/pc^2) of the sheet: all the mass within |z| < 1.1 kpc. */
  readonly sheetSurf: number
  readonly phi: (z: number) => number
}

function toyModel(mode: HaloMode): Model {
  const disk = mode === 'with' ? SOLAR_TOY : { ...SOLAR_TOY, rhoHalo: 0 }
  return {
    disk,
    sheetSurf: surfaceDensityWithin(disk, SHEET_LIMIT_KPC),
    phi: (z) => toyPotential(disk, z)
  }
}

/** The h_Z of the tracer and z_0 of the sheet over a grid of dispersions. */
interface ScalingPoint {
  readonly sigma: number
  readonly tracer: number
  readonly sheet: number
}

function scaling(model: Model): readonly ScalingPoint[] {
  return geomspace(SIGMA_RANGE[0], SIGMA_RANGE[1], 36).map((sigma) => ({
    sigma,
    tracer: scaledProfileHz(tracerProfile(model.phi, sigma)).hz,
    sheet: sheetScale(sigma, model.sheetSurf)
  }))
}

/** The local slope d ln h / d ln sigma of the tracer at `sigma`, from the two nearest points. */
function localSlope(points: readonly ScalingPoint[], sigma: number): number {
  const i = Math.min(
    Math.max(
      d3.bisectRight(
        points.map((p) => p.sigma),
        sigma
      ),
      1
    ),
    points.length - 1
  )
  const [a, b] = [points[i - 1], points[i]]
  return Math.log(b.tracer / a.tracer) / Math.log(b.sigma / a.sigma)
}

interface ProfilesProps {
  readonly model: Model
  readonly sigma: number
  readonly tracerHz: number
  readonly width: number
}

/** The whole disk, and the population as a tracer and as a sheet, each of unit mass. */
function Profiles({ model, sigma, tracerHz, width }: ProfilesProps): ReactNode {
  const zDisk = diskScale(model.disk)
  const z0 = sheetScale(sigma, model.sheetSurf)
  const zMax = Math.min(Math.max(3 * Math.max(tracerHz, z0), 2.5 * zDisk, 0.3), 12)
  const z = linspace(0, zMax, PROFILE_POINTS)
  const tracerFull = tracerProfile(model.phi, sigma)
  const tracerMass = trapezoid(tracerFull, Z_GRID)
  const tracer = tracerProfile(model.phi, sigma, z).map((rho) => rho / tracerMass)
  const sheet = z.map((height) => sech2(height, z0) / z0)
  const disk = z.map((height) => sech2(height, zDisk) / zDisk)
  const top = (d3.max([...tracer, ...sheet, ...disk]) ?? 1) * 1.08
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
        const curve = (values: readonly number[]) =>
          linePath(values.map((v, i) => [x(z[i]), y(v)] as const))
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
              <path
                d={bandPath(
                  z.map((height) => [x(height), y(0)] as const),
                  disk.map((v, i) => [x(z[i]), y(v)] as const)
                )}
                className="viz-area"
              />
              <path d={curve(sheet)} className={dashClass('dashed')} style={{ stroke: SHEET }} />
              <path d={curve(tracer)} className={dashClass('solid')} style={{ stroke: TRACER }} />
              <line
                x1={x(tracerHz)}
                x2={x(tracerHz)}
                y1={0}
                y2={inner.height}
                className="viz-line viz-line--thin"
                style={{ stroke: TRACER }}
              />
              <line
                x1={x(z0)}
                x2={x(z0)}
                y1={0}
                y2={inner.height}
                className="viz-line viz-line--thin viz-line--dashed"
                style={{ stroke: SHEET }}
              />
            </g>
          </>
        )
      }}
    </Chart>
  )
}

interface ScalingProps {
  readonly model: Model
  readonly points: readonly ScalingPoint[]
  readonly sigma: number
  readonly width: number
}

/** h_Z of the tracer and z_0 of the sheet against sigma_z, with the slopes 1 and 2 for reference. */
function Scaling({ model, points, sigma, width }: ScalingProps): ReactNode {
  const finite = points.filter((p) => Number.isFinite(p.tracer))
  const heights = [...finite.map((p) => p.tracer), ...points.map((p) => p.sheet)]
  const yDomain: readonly [number, number] = [
    (d3.min(heights) ?? 0.01) / 1.3,
    (d3.max(heights) ?? 10) * 1.3
  ]
  const tracerNow = Math.exp(
    d3.scaleLinear(
      finite.map((p) => Math.log(p.sigma)),
      finite.map((p) => Math.log(p.tracer))
    )(Math.log(sigma))
  )
  const sheetNow = sheetScale(sigma, model.sheetSurf)
  const sigmaDisk = model.disk.sigmaDisk
  return (
    <Chart width={width} height={SCALING_HEIGHT} xLabel="$\sigma_z$ (km/s)" yLabel="$h_Z$ (kpc)">
      {(inner) => {
        const x = d3.scaleLog().domain(SIGMA_RANGE).range([0, inner.width])
        const y = d3.scaleLog().domain(yDomain).range([inner.height, 0])
        // Guides of slope 1 and 2 through the tracer at the lowest dispersion.
        const anchor = finite[0]
        const guide = (power: number) =>
          linePath(
            SIGMA_RANGE.map((s) => [x(s), y(anchor.tracer * (s / anchor.sigma) ** power)] as const)
          )
        return (
          <>
            <Axis
              scale={y}
              orientation="left"
              tickValues={ticksWithin(HEIGHT_TICKS, yDomain)}
              tickFormat={(v) => String(v)}
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
              <path d={guide(1)} className="viz-guide viz-line--dotted" />
              <path d={guide(2)} className="viz-guide viz-line--dashed" />
              <line
                x1={x(sigmaDisk)}
                x2={x(sigmaDisk)}
                y1={0}
                y2={inner.height}
                className="viz-guide"
              />
              <text x={x(sigmaDisk) + 4} y={12} className="viz-text">
                the disk itself
              </text>
              <path
                d={linePath(points.map((p) => [x(p.sigma), y(p.sheet)] as const))}
                className={dashClass('dashed')}
                style={{ stroke: SHEET }}
              />
              <path
                d={linePath(finite.map((p) => [x(p.sigma), y(p.tracer)] as const))}
                className={dashClass('solid')}
                style={{ stroke: TRACER }}
              />
              <line
                x1={x(sigma)}
                x2={x(sigma)}
                y1={0}
                y2={inner.height}
                className="viz-line viz-line--thin"
              />
              <Marker shape="circle" x={x(sigma)} y={y(tracerNow)} color={TRACER} />
              <Marker shape="square" x={x(sigma)} y={y(sheetNow)} open color={SHEET} />
            </g>
          </>
        )
      }}
    </Chart>
  )
}

/**
 * The two models of the thickness of a population, in a toy disk: a self-gravitating layer of
 * 55 Msun/pc^2 at 18 km/s, with or without a dark halo of 0.012 Msun/pc^3 near the plane. A
 * slider sets the dispersion of the population; the charts show its profile as a tracer of the
 * whole potential and as a sheet that makes its own, and how the height of each grows with the
 * dispersion.
 */
export default function ModelsChart(): ReactNode {
  const [ref, width] = useElementWidth<HTMLDivElement>()
  const [sigma, setSigma] = useState(30)
  const [mode, setMode] = useState<HaloMode>('with')
  const model = useMemo(() => toyModel(mode), [mode])
  const points = useMemo(() => scaling(model), [model])
  const tracerHz = useMemo(
    () => scaledProfileHz(tracerProfile(model.phi, sigma)).hz,
    [model, sigma]
  )
  const z0 = sheetScale(sigma, model.sheetSurf)
  return (
    <div className="viz" ref={ref}>
      <Profiles model={model} sigma={sigma} tracerHz={tracerHz} width={width} />
      <Legend
        entries={[
          { label: 'the whole disk, whose mass makes the potential', band: true },
          { label: 'the population as a tracer', dash: 'solid', color: TRACER },
          { label: 'the population as a sheet', dash: 'dashed', color: SHEET }
        ]}
      />
      <Scaling model={model} points={points} sigma={sigma} width={width} />
      <Legend
        entries={[
          { label: 'tracer, $h_Z$', dash: 'solid', marker: 'circle', color: TRACER },
          { label: 'sheet, $z_0$', dash: 'dashed', marker: 'square', open: true, color: SHEET },
          { label: '$\\propto \\sigma_z$', dash: 'dotted' },
          { label: '$\\propto \\sigma_z^2$', dash: 'dashed' }
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
          label="Dark halo"
          options={[
            ['with', 'with a dark halo'],
            ['without', 'without one']
          ]}
          isPressed={(option) => option === mode}
          onToggle={setMode}
        />
      </div>
      <p className="viz-note">
        <MathLabel
          label={`At $\\sigma_z = ${sigma.toFixed(0)}$ km/s the tracer has $h_Z = ${tracerHz.toFixed(2)}$ kpc and the sheet $z_0 = ${z0.toFixed(2)}$ kpc, ${(z0 / tracerHz).toFixed(2)} times as thick. The tracer grows locally as $\\sigma_z^{${localSlope(points, sigma).toFixed(2)}}$, the sheet always as $\\sigma_z^2$.`}
        />
      </p>
    </div>
  )
}
