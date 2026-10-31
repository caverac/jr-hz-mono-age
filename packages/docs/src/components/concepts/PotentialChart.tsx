import HaloChips, { DEFAULT_HALO, haloOf } from '@site/src/components/concepts/HaloChips'
import {
  type ConceptsRecord,
  type SliceCheck,
  type Split,
  useConcepts,
  usePotentialCheck
} from '@site/src/components/concepts/record'
import MathLabel from '@site/src/components/MathLabel'
import { Chips, Pending } from '@site/src/components/status/controls'
import Chart, { Axis, linePath } from '@site/src/components/viz/Chart'
import { useElementWidth } from '@site/src/components/viz/hooks'
import { dashClass, type Dash, Legend, Marker, seriesColor } from '@site/src/components/viz/marks'
import * as d3 from 'd3'
import React, { type ReactNode, useId, useState } from 'react'

const LINE_HEIGHT = 240
/** The levels of the lines of equal log10 density (Msun/pc^3), the same for both expansions. */
const LEVELS = d3.range(-4.5, -0.99, 0.5)
/** A value below every level, which closes the lines inside the edges of the map. */
const FLOOR = -20
/** The top of the map (kpc): above it the density of the CylSpline approaches the edge of its grid, at 10 kpc. */
const MAP_TOP = 8
/** The pixels by which the lines stay inside the frame, where the padding closes them. */
const EDGE_INSET = 3
/** The radius (in cells) of the blur that smooths the noise of the densities before the lines are traced. */
const SMOOTHING = 1
const CYLSPLINE = seriesColor(0)
const MULTIPOLE = seriesColor(1)
/** The annulus of the mono-age populations (kpc). */
const ANNULUS: readonly [number, number] = [7.5, 9]
const MARGIN = { top: 12, right: 16, bottom: 32, left: 48 }
/** The gutter of the rotated label of the y axis, as in `viz/Chart`. */
const LABEL_GUTTER = 26

/** The three lines of a split quantity: the whole potential and its two expansions. */
const PARTS: readonly (readonly [keyof Split, string, Dash, string | undefined])[] = [
  ['total', 'whole fit', 'solid', undefined],
  ['cylspline', 'CylSpline: gas and stars of the disk region', 'dashed', CYLSPLINE],
  ['multipole', 'Multipole: dark matter and everything else', 'dotted', MULTIPOLE]
]

type GridShown = 'none' | 'cylspline' | 'multipole'

/** The lines of equal density of one expansion, on the map padded with FLOOR so that they close. */
function isodensity(
  record: ConceptsRecord,
  values: readonly (readonly number[])[]
): d3.ContourMultiPolygon[] {
  const width = record.mapR.length
  const height = record.mapZ.length
  const smoothed = new Array<number>(width * height)
  values.forEach((row, i) => row.forEach((v, j) => (smoothed[j * width + i] = v)))
  d3.blur2({ data: smoothed, width, height }, SMOOTHING)
  const nR = width + 2
  const nZ = height + 2
  const flat = new Array<number>(nR * nZ).fill(FLOOR)
  smoothed.forEach((v, k) => (flat[(Math.floor(k / width) + 1) * nR + ((k % width) + 1)] = v))
  return d3
    .contours()
    .size([nR, nZ])
    .thresholds([...LEVELS])(flat)
}

interface MapProps {
  readonly record: ConceptsRecord
  readonly grid: GridShown
  readonly width: number
}

/**
 * The lines of equal density of both expansions in the meridional plane, R and z on the same
 * scale, so that a round distribution draws circles; with the nodes of either grid if asked.
 */
function DensityMap({ record, grid, width }: MapProps): ReactNode {
  const insetId = `inset-${useId().replace(/[^a-zA-Z0-9]/g, '')}`
  const rMin = record.mapR[0]
  const rMax = record.mapR[record.mapR.length - 1]
  const zMax = MAP_TOP
  const innerWidth = width - LABEL_GUTTER - MARGIN.left - MARGIN.right
  const height = Math.round((innerWidth * zMax) / (rMax - rMin)) + MARGIN.top + MARGIN.bottom
  const dR = record.mapR[1] - record.mapR[0]
  const dZ = record.mapZ[1] - record.mapZ[0]
  return (
    <Chart width={width} height={height} margin={MARGIN} xLabel="$R$ (kpc)" yLabel="$z$ (kpc)">
      {(inner) => {
        const x = d3.scaleLinear().domain([rMin, rMax]).range([0, inner.width])
        const y = d3.scaleLinear().domain([0, zMax]).range([inner.height, 0])
        // A contour vertex at (px, py) lies on the grid of the padded map, one cell off each edge.
        const project = d3.geoTransform({
          point(px, py) {
            this.stream.point(
              x(record.mapR[0] + (px - 1.5) * dR),
              y(record.mapZ[0] + (py - 1.5) * dZ)
            )
          }
        })
        const path = d3.geoPath(project)
        const parts = [
          ['cylspline', isodensity(record, record.logCylspline), 'dashed', CYLSPLINE],
          ['multipole', isodensity(record, record.logMultipole), 'dotted', MULTIPOLE]
        ] as const
        return (
          <>
            <defs>
              <clipPath id={insetId}>
                <rect
                  x={EDGE_INSET}
                  y={EDGE_INSET}
                  width={inner.width - 2 * EDGE_INSET}
                  height={inner.height - 2 * EDGE_INSET}
                />
              </clipPath>
            </defs>
            <Axis scale={y} orientation="left" ticks={4} gridLength={inner.width} />
            <Axis
              scale={x}
              orientation="bottom"
              transform={`translate(0,${inner.height})`}
              ticks={6}
              gridLength={inner.height}
            />
            <g clipPath={inner.clip}>
              <rect
                x={x(ANNULUS[0])}
                y={0}
                width={x(ANNULUS[1]) - x(ANNULUS[0])}
                height={inner.height}
                className="viz-window"
              />
              {grid === 'cylspline'
                ? record.grids.cylsplineR
                    .filter((r) => r <= rMax)
                    .flatMap((r) =>
                      record.grids.cylsplineZ
                        .filter((z) => z <= zMax)
                        .map((z) => (
                          <circle
                            key={`${r}-${z}`}
                            cx={x(r)}
                            cy={y(z)}
                            r={1.1}
                            className="viz-node"
                          />
                        ))
                    )
                : null}
              {grid === 'multipole'
                ? record.grids.multipoleR
                    .filter((r) => r <= Math.hypot(rMax, zMax))
                    .map((r) => (
                      <path
                        key={r}
                        d={linePath(
                          d3
                            .range(0, 91)
                            .map(
                              (deg) =>
                                [
                                  x(r * Math.cos((deg * Math.PI) / 180)),
                                  y(r * Math.sin((deg * Math.PI) / 180))
                                ] as const
                            )
                        )}
                        className="viz-node"
                      />
                    ))
                : null}
              <g clipPath={`url(#${insetId})`}>
                {parts.map(([key, lines, dash, color]) =>
                  lines.map((line) => (
                    <path
                      key={`${key}-${line.value}`}
                      d={path(line) ?? ''}
                      className={dashClass(dash)}
                      style={{ stroke: color, fill: 'none' }}
                    />
                  ))
                )}
              </g>
              <line
                x1={x(record.rMid)}
                x2={x(record.rMid)}
                y1={0}
                y2={inner.height}
                className="viz-guide"
              />
            </g>
          </>
        )
      }}
    </Chart>
  )
}

/** The density of each expansion against height at R_mid, from the column of the maps nearest it. */
function DensityColumn({
  record,
  width
}: {
  readonly record: ConceptsRecord
  readonly width: number
}): ReactNode {
  const i = d3.minIndex(record.mapR, (r) => Math.abs(r - record.rMid))
  const columns = [
    ['cylspline', record.logCylspline[i], 'dashed', CYLSPLINE],
    ['multipole', record.logMultipole[i], 'dotted', MULTIPOLE]
  ] as const
  const values = columns.flatMap(([, column]) => column).filter((v) => v > -11)
  const yDomain: readonly [number, number] = [
    Math.max(d3.min(values) ?? -6, -6),
    (d3.max(values) ?? 0) + 0.3
  ]
  // Up to the heights of the vertical structure: near the top of the CylSpline grid its density is not reliable.
  const zTop = record.z[record.z.length - 1]
  return (
    <Chart
      width={width}
      height={LINE_HEIGHT}
      xLabel="$|z|$ (kpc) at $R_\mathrm{mid}$"
      yLabel="$\log_{10}\rho$ ($M_\odot$ pc$^{-3}$)"
    >
      {(inner) => {
        const x = d3.scaleLinear().domain([0, zTop]).range([0, inner.width])
        const y = d3.scaleLinear().domain(yDomain).range([inner.height, 0])
        return (
          <>
            <Axis scale={y} orientation="left" ticks={5} gridLength={inner.width} />
            <Axis
              scale={x}
              orientation="bottom"
              transform={`translate(0,${inner.height})`}
              ticks={6}
              gridLength={inner.height}
            />
            <g clipPath={inner.clip}>
              {columns.map(([key, column, dash, color]) => (
                <path
                  key={key}
                  d={linePath(column.map((v, j) => [x(record.mapZ[j]), y(v)] as const))}
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

interface SplitProps {
  readonly x: readonly number[]
  readonly split: Split
  readonly width: number
  readonly xLabel: string
  readonly yLabel: string
  readonly marker?: number
}

/** A quantity of the whole fit and of its two expansions against height or radius. */
function SplitChart({ x: xs, split, width, xLabel, yLabel, marker }: SplitProps): ReactNode {
  const all = [...split.total, ...split.cylspline, ...split.multipole].filter(Number.isFinite)
  const yDomain: readonly [number, number] = [
    Math.min(0, d3.min(all) ?? 0),
    (d3.max(all) ?? 1) * 1.05
  ]
  return (
    <Chart width={width} height={LINE_HEIGHT} xLabel={xLabel} yLabel={yLabel}>
      {(inner) => {
        const x = d3
          .scaleLinear()
          .domain([xs[0], xs[xs.length - 1]])
          .range([0, inner.width])
        const y = d3.scaleLinear().domain(yDomain).range([inner.height, 0])
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
              {marker === undefined ? null : (
                <line
                  x1={x(marker)}
                  x2={x(marker)}
                  y1={0}
                  y2={inner.height}
                  className="viz-guide"
                />
              )}
              {PARTS.map(([key, label, dash, color]) => (
                <path
                  key={label}
                  d={linePath(xs.map((v, i) => [x(v), y(split[key][i])] as const))}
                  className={dashClass(dash)}
                  style={color === undefined ? undefined : { stroke: color }}
                />
              ))}
            </g>
          </>
        )
      }}
    </Chart>
  )
}

/** The local power p = d ln Delta Phi / d ln z of the whole fit, between 2 (harmonic) and 1 (linear). */
function PowerChart({
  record,
  width
}: {
  readonly record: ConceptsRecord
  readonly width: number
}): ReactNode {
  const { z } = record
  const phi = record.phi.total
  const points = z
    .map((height, i) => {
      if (i === 0 || i === z.length - 1) {
        return null
      }
      const p = Math.log(phi[i + 1] / phi[i - 1]) / Math.log(z[i + 1] / z[i - 1])
      return Number.isFinite(p) ? ([height, p] as const) : null
    })
    .filter((point): point is readonly [number, number] => point !== null)
  return (
    <Chart
      width={width}
      height={LINE_HEIGHT}
      xLabel="$|z|$ (kpc)"
      yLabel="$p = d\ln\Delta\Phi/d\ln z$"
    >
      {(inner) => {
        const x = d3
          .scaleLinear()
          .domain([0, z[z.length - 1]])
          .range([0, inner.width])
        const y = d3.scaleLinear().domain([0.6, 2.2]).range([inner.height, 0])
        const level = (v: number) => `M0,${y(v)}H${inner.width}`
        return (
          <>
            <Axis scale={y} orientation="left" ticks={5} gridLength={inner.width} />
            <Axis
              scale={x}
              orientation="bottom"
              transform={`translate(0,${inner.height})`}
              ticks={6}
              gridLength={inner.height}
            />
            <g clipPath={inner.clip}>
              <path d={level(2)} className="viz-guide viz-line--dotted" />
              <path d={level(1)} className="viz-guide viz-line--dashed" />
              <text x={inner.width - 4} y={y(2) - 4} className="viz-text" textAnchor="end">
                harmonic: tracers grow with η = 1
              </text>
              <text x={inner.width - 4} y={y(1) - 4} className="viz-text" textAnchor="end">
                linear: tracers grow with η = 2
              </text>
              <path
                d={linePath(points.map(([h, p]) => [x(h), y(p)] as const))}
                className="viz-line"
              />
            </g>
          </>
        )
      }}
    </Chart>
  )
}

/** The median Delta Phi of the snapshot and of the fit in the |z| slices of the annulus. */
function CheckChart({
  slices,
  width
}: {
  readonly slices: readonly SliceCheck[]
  readonly width: number
}): ReactNode {
  const top = (d3.max(slices.flatMap((s) => [s.snapshot, s.fitted])) ?? 1) * 1.08
  const zMax = (d3.max(slices.map((s) => s.z)) ?? 1.2) * 1.05
  return (
    <Chart
      width={width}
      height={LINE_HEIGHT}
      xLabel="$|z|$ (kpc)"
      yLabel="$\Delta\Phi$ (km$^2$ s$^{-2}$)"
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
            <path
              d={linePath(slices.map((s) => [x(s.z), y(s.fitted)] as const))}
              className="viz-line"
            />
            {slices.map((s) => (
              <Marker
                key={s.z}
                shape="circle"
                open
                x={x(s.z)}
                y={y(s.snapshot)}
                color={MULTIPOLE}
              />
            ))}
          </>
        )
      }}
    </Chart>
  )
}

/**
 * The fitted potential of one halo, expansion by expansion: where each holds its mass, with the
 * nodes of its grid; the vertical force, the local power of Delta Phi and the circular velocity
 * of each; and the fit against the potential that the snapshot stores for its star particles.
 */
export default function PotentialChart(): ReactNode {
  const [ref, width] = useElementWidth<HTMLDivElement>()
  const [label, setLabel] = useState(DEFAULT_HALO)
  const [grid, setGrid] = useState<GridShown>('none')
  const halo = haloOf(label)
  const { value, error } = useConcepts(halo)
  const check = usePotentialCheck(halo)
  return (
    <div className="viz" ref={ref}>
      <div className="viz-controls" style={{ borderTop: 'none', marginTop: 0, paddingTop: 0 }}>
        <HaloChips selected={label} onSelect={setLabel} />
        <Chips
          label="Grid"
          options={[
            ['none', 'no grid'],
            ['cylspline', 'CylSpline nodes'],
            ['multipole', 'Multipole radii']
          ]}
          isPressed={(option) => option === grid}
          onToggle={setGrid}
        />
      </div>
      {value === null ? (
        <Pending error={error} />
      ) : (
        <>
          <DensityMap record={value} grid={grid} width={width} />
          <Legend
            entries={[
              {
                label: 'CylSpline: gas and stars of the disk region',
                dash: 'dashed',
                color: CYLSPLINE
              },
              {
                label: 'Multipole: dark matter and everything else',
                dash: 'dotted',
                color: MULTIPOLE
              },
              { label: 'the annulus of the populations', band: true }
            ]}
          />
          <p className="viz-note">
            <MathLabel
              label={`Lines of equal density, $\\log_{10}\\rho$ from $-4.5$ to $-1$ ($M_\\odot$ pc$^{-3}$) in steps of 0.5, traced on the densities lightly smoothed over one cell of the map, with $R$ and $z$ on the same scale up to $z = 8$ kpc. The vertical line is the median radius of the annulus, $R_\\mathrm{mid} = ${value.rMid.toFixed(2)}$ kpc. The CylSpline has ${value.grids.cylsplineR.length} radii and ${value.grids.cylsplineZ.length} heights, the Multipole ${value.grids.multipoleR.length} radii and $l_\\mathrm{max} = ${value.grids.multipoleLmax}$.`}
            />
          </p>
          <DensityColumn record={value} width={width} />
          <SplitChart
            x={value.z}
            split={value.force}
            width={width}
            xLabel="$|z|$ (kpc) at $R_\mathrm{mid}$"
            yLabel="$K_z$ (km$^2$ s$^{-2}$ kpc$^{-1}$)"
          />
          <PowerChart record={value} width={width} />
          <SplitChart
            x={value.r}
            split={value.vCirc}
            width={width}
            xLabel="$R$ (kpc)"
            yLabel="$v_c$ (km/s)"
            marker={value.rMid}
          />
          <Legend
            entries={PARTS.map(([, partLabel, dash, color]) => ({ label: partLabel, dash, color }))}
          />
          {check.value === null ? (
            <Pending error={check.error} />
          ) : (
            <>
              <CheckChart slices={check.value} width={width} />
              <Legend
                entries={[
                  {
                    label: 'the fitted potential, median over the stars of each slice',
                    dash: 'solid'
                  },
                  {
                    label: 'the potential the snapshot stores, median over the same stars',
                    marker: 'circle',
                    open: true,
                    color: MULTIPOLE
                  }
                ]}
              />
            </>
          )}
        </>
      )}
    </div>
  )
}
