import MathLabel from '@site/src/components/MathLabel'
import { Chips, Pending, toggled } from '@site/src/components/status/controls'
import { plusMinus } from '@site/src/components/status/format'
import { AGREEMENT, median } from '@site/src/components/status/metrics'
import {
  type ImigCell,
  type ImigCells,
  isMeasured,
  MILKY_WAY_PATH,
  parseMilkyWay,
  type TracerEstimator,
  tracerGrowth
} from '@site/src/components/status/milkyWay'
import useRecord from '@site/src/components/status/useRecord'
import Chart, { Axis, linePath, ticksWithin } from '@site/src/components/viz/Chart'
import { useElementWidth, useRevealed } from '@site/src/components/viz/hooks'
import {
  Legend,
  type LegendEntry,
  Marker,
  type MarkerShape,
  Swatch
} from '@site/src/components/viz/marks'
import * as d3 from 'd3'
import React, { type ReactNode, useState } from 'react'

const HEIGHTS_PANEL = 330
const RATIO_PANEL = 190
const SIGMA_TICKS = [8, 10, 15, 20, 25, 30]
const AGE_TICKS = [1, 2, 3, 5, 7, 10]
const HEIGHT_TICKS = [0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1]
const RATIO_TICKS = [0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8, 2]
/** The part of the logarithmic width of an age bin over which its cells spread by [M/H]. */
const SPREAD: readonly [number, number] = [0.18, 0.82]
/**
 * Where the segment with the slope of a harmonic potential sits: this factor below the power
 * law of our counts, from the sigma_z of this quantile of the cells to the largest. It shows
 * a slope, not a fit.
 */
const GUIDE = { offset: 0.6, quantile: 0.6 }

/** What the shared horizontal axis shows. */
type Abscissa = 'sigma' | 'age'

const ABSCISSAE: readonly (readonly [Abscissa, string])[] = [
  ['sigma', 'against $\\sigma_z$'],
  ['age', 'against $\\tau$']
]

/** The metallicity groups of the cells, by the center of their [M/H] bin, as the prose splits them. */
type Group = 'poor' | 'solar' | 'rich'

const GROUPS: readonly {
  readonly group: Group
  readonly label: string
  readonly marker: MarkerShape
}[] = [
  { group: 'poor', label: '$[\\mathrm{M/H}] < -0.1$', marker: 'circle' },
  { group: 'solar', label: '$-0.1 \\le [\\mathrm{M/H}] < 0.1$', marker: 'square' },
  { group: 'rich', label: '$[\\mathrm{M/H}] \\ge 0.1$', marker: 'triangle' }
]

function groupColor(group: Group): string {
  return `var(--viz-metal-${group})`
}

function metallicity(cell: ImigCell): number {
  return 0.5 * (cell.mhLo + cell.mhHi)
}

function groupOf(cell: ImigCell): Group {
  const mh = metallicity(cell)
  return mh < -0.1 ? 'poor' : mh < 0.1 ? 'solar' : 'rich'
}

function markerOf(group: Group): MarkerShape {
  return GROUPS.find((g) => g.group === group)?.marker ?? 'circle'
}

function ageKey(cell: ImigCell): string {
  return `${cell.ageLo}-${cell.ageHi}`
}

function cellKey(cell: ImigCell): string {
  return `${ageKey(cell)}/${cell.mhLo}`
}

function isPositive(value: number): boolean {
  return Number.isFinite(value) && value > 0
}

/** A value of the cell with its error, as the panels draw it. */
type Read = (cell: ImigCell) => { readonly value: number; readonly err: number }

/** The two ratios of the lower panels, from the top. */
const RATIOS: readonly { readonly read: Read; readonly label: string }[] = [
  { read: (c) => c.ratios.exponential, label: '$h_\\mathrm{exp}/h_\\mathrm{exp,I}$' },
  { read: (c) => c.ratios.tracer, label: '$h_\\mathrm{tracer}/h_Z$' }
]

/** The power law h = h_0 (sigma / sigma_0)^slope through the weighted mean of ln h at ln sigma. */
function powerLaw(
  cells: readonly ImigCell[],
  read: Read,
  slope: number
): (sigma: number) => number {
  const points = cells.map((c) => ({
    x: Math.log(c.kinematics.sigmaZ.value),
    y: Math.log(read(c).value),
    w: (read(c).value / read(c).err) ** 2
  }))
  const total = d3.sum(points, (p) => p.w)
  const x0 = d3.sum(points, (p) => p.w * p.x) / total
  const y0 = d3.sum(points, (p) => p.w * p.y) / total
  return (sigma) => Math.exp(y0 + slope * (Math.log(sigma) - x0))
}

/** The horizontal scale of both abscissae, and where a cell sits on it. */
interface Abscissae {
  readonly domain: readonly [number, number]
  readonly ticks: readonly number[]
  readonly label: string
  readonly at: (cell: ImigCell) => number
  /** The age bins of the grid, shaded alternately against age. */
  readonly bins: readonly { readonly lo: number; readonly hi: number }[]
}

function abscissae(abscissa: Abscissa, all: ImigCells, shown: readonly ImigCell[]): Abscissae {
  if (abscissa === 'sigma') {
    const sigmas = shown.map((c) => c.kinematics.sigmaZ.value)
    return {
      domain: [(d3.min(sigmas) ?? 10) / 1.1, (d3.max(sigmas) ?? 25) * 1.15],
      ticks: SIGMA_TICKS,
      label: '$\\sigma_z$ (km/s)',
      at: (c) => c.kinematics.sigmaZ.value,
      bins: []
    }
  }
  const bins = d3.sort(
    d3.groups(all.cells, ageKey).map(([, g]) => ({ lo: g[0].ageLo, hi: g[0].ageHi })),
    (b) => b.lo
  )
  const mh = d3.extent(all.cells, metallicity)
  const spread = d3
    .scaleLinear()
    .domain([mh[0] ?? -0.5, mh[1] ?? 0.5])
    .range(SPREAD)
  return {
    domain: [d3.min(shown, (c) => c.ageLo) ?? 1, d3.max(shown, (c) => c.ageHi) ?? 10],
    ticks: AGE_TICKS,
    label: '$\\tau$ (Gyr)',
    at: (c) => c.ageLo * (c.ageHi / c.ageLo) ** spread(metallicity(c)),
    bins
  }
}

interface PanelProps {
  readonly height: number
  readonly width: number
  readonly yLabel: string
  readonly yDomain: readonly [number, number]
  readonly yTicks: readonly number[]
  readonly x: Abscissae
  readonly last: boolean
  readonly revealed: boolean
  readonly draw: (
    x: d3.ScaleLogarithmic<number, number>,
    y: d3.ScaleLogarithmic<number, number>,
    innerWidth: number
  ) => ReactNode
}

/** One panel on the shared abscissa, with its axes and the shading of the age bins. */
function Panel({
  height,
  width,
  yLabel,
  yDomain,
  yTicks,
  x: abscissa,
  last,
  revealed,
  draw
}: PanelProps): ReactNode {
  return (
    <Chart width={width} height={height} xLabel={last ? abscissa.label : undefined} yLabel={yLabel}>
      {(inner) => {
        const x = d3.scaleLog().domain(abscissa.domain).range([0, inner.width])
        const y = d3.scaleLog().domain(yDomain).range([inner.height, 0])
        return (
          <g className={revealed ? 'viz-reveal viz-reveal--in' : 'viz-reveal'}>
            <g clipPath={inner.clip}>
              {abscissa.bins.map((b, i) =>
                i % 2 === 0 ? (
                  <rect
                    key={b.lo}
                    x={x(b.lo)}
                    width={x(b.hi) - x(b.lo)}
                    y={0}
                    height={inner.height}
                    className="viz-band viz-fadein"
                  />
                ) : null
              )}
            </g>
            <Axis
              scale={y}
              orientation="left"
              tickValues={ticksWithin(yTicks, yDomain)}
              tickFormat={(v) => String(v)}
              gridLength={inner.width}
            />
            <Axis
              scale={x}
              orientation="bottom"
              transform={`translate(0,${inner.height})`}
              tickValues={ticksWithin(abscissa.ticks, abscissa.domain)}
              tickFormat={(v) => String(v)}
            />
            <g clipPath={inner.clip}>{draw(x, y, inner.width)}</g>
          </g>
        )
      }}
    </Chart>
  )
}

/** The vertical error bar of `value` at x = 0, in a group that the caller places. */
function ErrorBar({
  value,
  err,
  y,
  domain,
  color
}: {
  readonly value: number
  readonly err: number
  readonly y: d3.ScaleLogarithmic<number, number>
  readonly domain: readonly [number, number]
  readonly color: string
}): ReactNode {
  return (
    <line
      x1={0}
      x2={0}
      y1={y(Math.max(value - err, domain[0]))}
      y2={y(Math.min(value + err, domain[1]))}
      className="viz-line viz-line--thin viz-errorbar"
      style={{ stroke: color }}
    />
  )
}

interface Shared {
  /** Every measured cell, which sets the scales and the power laws, so that hiding a group moves nothing. */
  readonly measured: readonly ImigCell[]
  /** The measured cells of the groups shown. */
  readonly cells: readonly ImigCell[]
  readonly hovered: ImigCell | null
  readonly onHover: (cell: ImigCell | null) => void
}

/** A cell's group, moved along the abscissa when it changes, with its markers inside. */
function Placed({
  px,
  delay,
  children
}: {
  readonly px: number
  readonly delay: number
  readonly children: ReactNode
}): ReactNode {
  return (
    <g className="viz-move" style={{ transform: `translate(${px}px, 0px)` }}>
      <g className="viz-pop" style={{ animationDelay: `${delay}ms` }}>
        {children}
      </g>
    </g>
  )
}

function Ring({ cy }: { readonly cy: number }): ReactNode {
  return <circle cx={0} cy={cy} r={9} className="viz-line" />
}

/** The medians of `read` in the age bins of the cells, against age only. */
function Medians({
  cells,
  read,
  x,
  y
}: {
  readonly cells: readonly ImigCell[]
  readonly read: Read
  readonly x: d3.ScaleLogarithmic<number, number>
  readonly y: d3.ScaleLogarithmic<number, number>
}): ReactNode {
  const bins = d3.sort(
    d3.groups(cells, ageKey).map(([, g]) => ({
      age: Math.sqrt(g[0].ageLo * g[0].ageHi),
      median: median(g.map((c) => read(c).value))
    })),
    (b) => b.age
  )
  return (
    <path
      d={linePath(bins.map((b) => [x(b.age), y(b.median)] as const))}
      className="viz-line viz-line--median viz-fadein"
    />
  )
}

interface ChartData {
  readonly all: ImigCells
  readonly estimators: readonly TracerEstimator[]
}

/** Panel a: our exponential h_Z and the published one of every cell, with the tracer and the power laws. */
function HeightsPanel({
  data,
  abscissa,
  x: axis,
  width,
  revealed,
  shared
}: {
  readonly data: ChartData
  readonly abscissa: Abscissa
  readonly x: Abscissae
  readonly width: number
  readonly revealed: boolean
  readonly shared: Shared
}): ReactNode {
  const { measured, cells, hovered, onHover } = shared
  const ours: Read = (c) => c.ours.exponential
  const theirs: Read = (c) => c.theirs.sun
  const heights = measured.flatMap((c) => [ours(c).value, theirs(c).value])
  const yDomain: readonly [number, number] = [
    (d3.min(heights) ?? 0.1) / 1.25,
    (d3.max(heights) ?? 1) * 1.25
  ]
  const { dispersion } = data.all
  const oursLaw = powerLaw(measured, ours, dispersion.countsExponential.value)
  const theirsLaw = powerLaw(measured, theirs, dispersion.imigSun.value)
  const sigmas = d3.sort(measured.map((c) => c.kinematics.sigmaZ.value))
  return (
    <Panel
      height={HEIGHTS_PANEL}
      width={width}
      yLabel="$h_Z$ (kpc)"
      yDomain={yDomain}
      yTicks={HEIGHT_TICKS}
      x={axis}
      last={false}
      revealed={revealed}
      draw={(x, y) => {
        const law = (f: (s: number) => number) =>
          linePath(axis.domain.map((s) => [x(s), y(f(s))] as const))
        const guideFrom = d3.quantile(sigmas, GUIDE.quantile) ?? axis.domain[0]
        const guideTo = sigmas[sigmas.length - 1] ?? axis.domain[1]
        const guideAt = (s: number) => GUIDE.offset * oursLaw(guideFrom) * (s / guideFrom)
        return (
          <>
            {abscissa === 'sigma' ? (
              <g className="viz-fadein">
                <path d={law(oursLaw)} className="viz-line viz-line--thin viz-line--dashed" />
                <path d={law(theirsLaw)} className="viz-line viz-line--thin viz-line--dashdot" />
                <path
                  d={linePath([guideFrom, guideTo].map((s) => [x(s), y(guideAt(s))] as const))}
                  className="viz-guide viz-line--dotted"
                />
                <path
                  d={linePath(data.estimators.map((e) => [x(e.sigmaZ), y(e.exponential)] as const))}
                  pathLength={1}
                  className="viz-line viz-line--median viz-draw"
                />
              </g>
            ) : null}
            {cells.map((c, i) => {
              const group = groupOf(c)
              const color = groupColor(group)
              const marker = markerOf(group)
              const hover = {
                onMouseEnter: () => onHover(c),
                onMouseLeave: () => onHover(null)
              }
              return (
                <Placed key={cellKey(c)} px={x(axis.at(c))} delay={i * 18}>
                  <ErrorBar {...ours(c)} y={y} domain={yDomain} color={color} />
                  <ErrorBar {...theirs(c)} y={y} domain={yDomain} color={color} />
                  {hovered !== null && cellKey(hovered) === cellKey(c) ? (
                    <>
                      <Ring cy={y(ours(c).value)} />
                      <Ring cy={y(theirs(c).value)} />
                    </>
                  ) : null}
                  <Marker shape={marker} color={color} x={0} y={y(ours(c).value)} {...hover} />
                  <Marker
                    shape={marker}
                    open
                    color={color}
                    x={0}
                    y={y(theirs(c).value)}
                    {...hover}
                  />
                </Placed>
              )
            })}
          </>
        )
      }}
    />
  )
}

/** Panels b and c: a ratio of every cell, with a band of agreement around 1. */
function RatioPanel({
  read,
  label,
  last,
  abscissa,
  x: axis,
  width,
  revealed,
  delay,
  shared
}: {
  readonly read: Read
  readonly label: string
  readonly last: boolean
  readonly abscissa: Abscissa
  readonly x: Abscissae
  readonly width: number
  readonly revealed: boolean
  readonly delay: number
  readonly shared: Shared
}): ReactNode {
  const { hovered, onHover } = shared
  const cells = shared.cells.filter((c) => isPositive(read(c).value))
  const values = shared.measured.map((c) => read(c).value).filter(isPositive)
  const yDomain: readonly [number, number] = [
    Math.min(Math.exp(-2 * AGREEMENT), d3.min(values) ?? 1) / 1.12,
    Math.max(Math.exp(2 * AGREEMENT), d3.max(values) ?? 1) * 1.12
  ]
  return (
    <Panel
      height={RATIO_PANEL}
      width={width}
      yLabel={label}
      yDomain={yDomain}
      yTicks={RATIO_TICKS}
      x={axis}
      last={last}
      revealed={revealed}
      draw={(x, y, innerWidth) => (
        <>
          <rect
            x={0}
            width={innerWidth}
            y={y(Math.exp(AGREEMENT))}
            height={y(Math.exp(-AGREEMENT)) - y(Math.exp(AGREEMENT))}
            className="viz-band"
          />
          <line x1={0} x2={innerWidth} y1={y(1)} y2={y(1)} className="viz-line viz-line--thin" />
          {cells.map((c, i) => {
            const group = groupOf(c)
            const color = groupColor(group)
            return (
              <Placed key={cellKey(c)} px={x(axis.at(c))} delay={delay + i * 18}>
                <ErrorBar {...read(c)} y={y} domain={yDomain} color={color} />
                {hovered !== null && cellKey(hovered) === cellKey(c) ? (
                  <Ring cy={y(read(c).value)} />
                ) : null}
                <Marker
                  shape={markerOf(group)}
                  color={color}
                  x={0}
                  y={y(read(c).value)}
                  onMouseEnter={() => onHover(c)}
                  onMouseLeave={() => onHover(null)}
                />
              </Placed>
            )
          })}
          {abscissa === 'age' ? <Medians cells={cells} read={read} x={x} y={y} /> : null}
        </>
      )}
    />
  )
}

function slopeText(value: number, err: number): string {
  return `$\\sigma_z^{${value.toFixed(2)} \\pm ${err.toFixed(2)}}$`
}

function Readout({
  data,
  abscissa,
  hovered
}: {
  readonly data: ChartData
  readonly abscissa: Abscissa
  readonly hovered: ImigCell | null
}): ReactNode {
  if (hovered === null) {
    const { dispersion, trends } = data.all
    const tracer = tracerGrowth(data.estimators)
    const text =
      abscissa === 'sigma'
        ? `At fixed [M/H], our exponential heights grow as ${slopeText(dispersion.countsExponential.value, dispersion.countsExponential.err)}, whereas the published ones grow as ${slopeText(dispersion.imigSun.value, dispersion.imigSun.err)} and the tracer as $\\sigma_z^{${tracer.slope.toFixed(2)}}$ from ${tracer.lo} to ${tracer.hi} km/s.`
        : `At fixed [M/H], our exponential fits over theirs change by ${plusMinus(trends.exponential.age.value, trends.exponential.age.err, 2)} per ln age, whereas the tracer over our counts changes by ${plusMinus(trends.tracer.age.value, trends.tracer.age.err, 2)}.`
    return (
      <p className="viz-note">
        <MathLabel label={`${text} Point at a cell to read its heights.`} />
      </p>
    )
  }
  const c = hovered
  const value = (e: { readonly value: number; readonly err: number }, digits: number) =>
    `$${e.value.toFixed(digits)} \\pm ${e.err.toFixed(digits)}$`
  return (
    <p className="viz-note">
      <Swatch color={groupColor(groupOf(c))} />
      <MathLabel
        label={`${c.ageLo.toFixed(2)}-${c.ageHi.toFixed(2)} Gyr, $[\\mathrm{M/H}]$ from $${c.mhLo.toFixed(1)}$ to $${c.mhHi.toFixed(1)}$: $\\sigma_z =$ ${value(c.kinematics.sigmaZ, 1)} km/s, our $h_\\mathrm{exp} =$ ${value(c.ours.exponential, 3)} kpc, their published $h_Z =$ ${value(c.theirs.sun, 3)} kpc, ${RATIOS.map((r) => `${r.label} $=$ ${value(r.read(c), 2)}`).join(', ')}.`}
      />
    </p>
  )
}

/**
 * The cells of age and [M/H] of Imig et al. (2025) that our counts measure and that have a
 * sigma_z, on a shared abscissa that the reader switches between sigma_z and age: our
 * exponential h_Z and their published one with the tracer and power laws (top), our
 * exponential fit over theirs through the same counts (middle), and the tracer over our
 * windowed h_Z (bottom). The cells glide to their new places when the abscissa changes.
 */
export default function DiscrepancyChart(): ReactNode {
  const [ref, width] = useElementWidth<HTMLDivElement>()
  const revealed = useRevealed(ref)
  const { value, error } = useRecord(MILKY_WAY_PATH, parseMilkyWay)
  const [abscissa, setAbscissa] = useState<Abscissa>('sigma')
  const [shown, setShown] = useState<ReadonlySet<Group>>(new Set(GROUPS.map((g) => g.group)))
  const [hovered, setHovered] = useState<ImigCell | null>(null)
  if (value === null) {
    return (
      <div className="viz" ref={ref}>
        <Pending error={error} />
      </div>
    )
  }
  const all = value.imigCells
  const measured = all.cells.filter(
    (c) =>
      isMeasured(all, c) &&
      isPositive(c.kinematics.sigmaZ.value) &&
      isPositive(c.ours.exponential.value) &&
      isPositive(c.theirs.sun.value)
  )
  const cells = measured.filter((c) => shown.has(groupOf(c)))
  const data: ChartData = { all, estimators: value.estimators }
  const x = abscissae(abscissa, all, measured)
  const shared: Shared = { measured, cells, hovered, onHover: setHovered }
  const { dispersion } = all
  const entries: readonly LegendEntry[] = [
    { label: 'our counts', marker: 'circle' },
    { label: 'Imig et al. (2025)', marker: 'circle', open: true },
    ...GROUPS.filter((g) => shown.has(g.group)).map((g) => ({
      label: g.label,
      marker: g.marker,
      color: groupColor(g.group)
    })),
    ...(abscissa === 'sigma'
      ? [
          { label: 'tracer', dash: 'solid' as const },
          {
            label: `$\\propto \\sigma_z^{${dispersion.countsExponential.value.toFixed(2)}}$`,
            dash: 'dashed' as const
          },
          {
            label: `$\\propto \\sigma_z^{${dispersion.imigSun.value.toFixed(2)}}$`,
            dash: 'dashdot' as const
          },
          { label: '$\\propto \\sigma_z$', dash: 'dotted' as const, color: 'var(--viz-muted)' }
        ]
      : [{ label: 'median of the age bin', dash: 'solid' as const }]),
    { label: `$|\\ln r| \\le ${AGREEMENT}$ for a ratio $r$`, band: true }
  ]
  return (
    <div className="viz" ref={ref}>
      <HeightsPanel
        data={data}
        abscissa={abscissa}
        x={x}
        width={width}
        revealed={revealed}
        shared={shared}
      />
      {RATIOS.map((r, i) => (
        <RatioPanel
          key={r.label}
          read={r.read}
          label={r.label}
          last={i === RATIOS.length - 1}
          abscissa={abscissa}
          x={x}
          width={width}
          revealed={revealed}
          delay={500 + i * 400}
          shared={shared}
        />
      ))}
      <Legend entries={entries} />
      <div className="viz-controls">
        <Chips
          label="Abscissa"
          options={ABSCISSAE}
          isPressed={(a) => a === abscissa}
          onToggle={setAbscissa}
        />
        <Chips
          label="Metallicity"
          options={GROUPS.map((g) => [g.group, g.label] as const)}
          isPressed={(group) => shown.has(group)}
          onToggle={(group) => setShown((current) => toggled(current, group))}
          swatch={groupColor}
        />
      </div>
      <Readout data={data} abscissa={abscissa} hovered={hovered} />
    </div>
  )
}
