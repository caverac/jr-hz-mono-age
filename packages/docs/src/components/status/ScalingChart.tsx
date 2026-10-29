import MathLabel from '@site/src/components/MathLabel'
import { haloColor } from '@site/src/components/render/halos'
import {
  fadeClass,
  HaloPicker,
  type OtherGroup,
  Pending
} from '@site/src/components/status/controls'
import type { Estimate, StatusData } from '@site/src/components/status/data'
import { ageRange } from '@site/src/components/status/metrics'
import {
  MILKY_WAY_PATH,
  type MilkyWayData,
  parseMilkyWay,
  surveyName
} from '@site/src/components/status/milkyWay'
import useRecord from '@site/src/components/status/useRecord'
import useStatusData from '@site/src/components/status/useStatusData'
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

const HEIGHT = 440
const SIGMA_TICKS = [5, 7, 10, 15, 20, 30, 40, 50, 70, 100]
/** The ticks of an x axis narrower than NARROW px, where the full set would collide. */
const SPARSE_SIGMA_TICKS = [5, 10, 20, 50, 100]
const NARROW = 380
const HEIGHT_TICKS = [0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1, 1.5, 2, 3, 5]
/** The range of ages (Gyr) whose eta the readout quotes. */
const ETA_RANGE: readonly [number, number] = [0, 10]
const MILKY_WAY = 'Milky Way'
const MILKY_WAY_GROUP: OtherGroup = { label: MILKY_WAY, color: 'var(--viz-fg)' }

/** The marker of every survey of the Milky Way; the Auriga populations are circles. */
const SURVEY_MARKERS: Readonly<Record<string, MarkerShape>> = {
  lamost: 'square',
  apogee: 'triangle',
  apogee_distmass: 'diamond'
}

/** One population: its sigma_z (km/s), its measured h_Z and the h_Z of its tracer (kpc). */
interface Population {
  readonly label: string
  readonly sigmaZ: Estimate
  readonly measured: Estimate
  readonly tracer: number
}

/** The populations of one halo or of one survey of the Milky Way, and how they are drawn. */
interface Group {
  readonly key: string
  /** The halo, or the Milky Way, that the picker highlights. */
  readonly galaxy: string
  readonly color: string
  readonly marker: MarkerShape
  /** In order of sigma_z, so that the line of the tracer runs through them in that order. */
  readonly populations: readonly Population[]
}

function isDrawn(p: Population): boolean {
  return [p.sigmaZ.value, p.measured.value, p.tracer].every((v) => Number.isFinite(v) && v > 0)
}

function bySigma(populations: readonly Population[]): readonly Population[] {
  return populations.filter(isDrawn).sort((a, b) => a.sigmaZ.value - b.sigmaZ.value)
}

function groupsOf(auriga: StatusData, milkyWay: MilkyWayData): readonly Group[] {
  const halos = auriga.halos.map((halo): Group => ({
    key: halo.label,
    galaxy: halo.label,
    color: haloColor(halo.label),
    marker: 'circle',
    populations: bySigma(
      halo.bins.map((bin) => ({
        label: `${halo.label}, ${bin.ageLo}-${bin.ageHi} Gyr`,
        sigmaZ: bin.sigmaZ,
        measured: bin.measured,
        tracer: bin.tracer
      }))
    )
  }))
  const surveys = milkyWay.surveys.map((survey): Group => ({
    key: survey.name,
    galaxy: MILKY_WAY,
    color: MILKY_WAY_GROUP.color,
    marker: SURVEY_MARKERS[survey.name] ?? 'square',
    populations: bySigma(
      survey.bins.map((bin) => ({
        label: `${surveyName(survey.name)}, ${bin.ageLo}-${bin.ageHi} Gyr`,
        sigmaZ: bin.sigmaZ,
        measured: bin.measured,
        tracer: bin.tracer
      }))
    )
  }))
  return [...halos, ...surveys]
}

/** exp(median(ln v)): the center of a set of positive values on a logarithmic axis. */
function logCenter(values: readonly number[]): number {
  return Math.exp(d3.median(values.map(Math.log)) ?? 0)
}

/** The domain of a logarithmic axis that holds `values`, padded by the factor `pad`. */
function logDomain(values: readonly number[], pad: number): readonly [number, number] {
  return [(d3.min(values) ?? 1) / pad, (d3.max(values) ?? 1) * pad]
}

interface Hovered {
  readonly group: Group
  readonly population: Population
}

interface PlotProps {
  readonly groups: readonly Group[]
  readonly width: number
  readonly selected: string | null
  readonly revealed: boolean
  readonly onHover: (hovered: Hovered | null) => void
}

function Plot({ groups, width, selected, revealed, onHover }: PlotProps): ReactNode {
  const all = groups.flatMap((g) => g.populations)
  const xDomain = logDomain(
    all.map((p) => p.sigmaZ.value),
    1.12
  )
  const yDomain = logDomain(
    all.flatMap((p) => [p.measured.value, p.tracer]),
    1.15
  )
  const auriga = groups.filter((g) => g.galaxy !== MILKY_WAY).flatMap((g) => g.populations)
  const pivot = {
    sigma: logCenter(auriga.map((p) => p.sigmaZ.value)),
    height: logCenter(auriga.map((p) => p.measured.value))
  }
  /** The order in which the markers enter, from the coldest population to the hottest. */
  const order = new Map(
    [...all].sort((a, b) => a.sigmaZ.value - b.sigmaZ.value).map((p, i) => [p, i] as const)
  )
  return (
    <Chart width={width} height={HEIGHT} xLabel="$\sigma_z$ (km/s)" yLabel="$h_Z$ (kpc)">
      {(inner) => {
        const x = d3.scaleLog().domain(xDomain).range([0, inner.width])
        const y = d3.scaleLog().domain(yDomain).range([inner.height, 0])
        const guide = (eta: number) =>
          linePath(xDomain.map((s) => [x(s), y(pivot.height * (s / pivot.sigma) ** eta)] as const))
        return (
          <g className={revealed ? 'viz-reveal viz-reveal--in' : 'viz-reveal'}>
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
              tickValues={ticksWithin(
                inner.width < NARROW ? SPARSE_SIGMA_TICKS : SIGMA_TICKS,
                xDomain
              )}
              tickFormat={(v) => String(v)}
              gridLength={inner.height}
            />
            <g clipPath={inner.clip} className="viz-fadein">
              <path d={guide(1)} className="viz-guide viz-line--dotted" />
              <path d={guide(2)} className="viz-guide viz-line--dashed" />
            </g>
            {groups.map((g, gi) => (
              <g key={g.key} className={fadeClass(selected, g.galaxy)}>
                <path
                  d={linePath(g.populations.map((p) => [x(p.sigmaZ.value), y(p.tracer)] as const))}
                  pathLength={1}
                  className="viz-line viz-draw"
                  style={{ stroke: g.color, animationDelay: `${200 + gi * 90}ms` }}
                />
                {g.populations.map((p) => {
                  const cx = x(p.sigmaZ.value)
                  const cy = y(p.measured.value)
                  return (
                    <g
                      key={p.label}
                      className="viz-pop"
                      style={{ animationDelay: `${(order.get(p) ?? 0) * 12}ms` }}
                    >
                      <line
                        x1={x(Math.max(p.sigmaZ.value - p.sigmaZ.err, xDomain[0]))}
                        x2={x(p.sigmaZ.value + p.sigmaZ.err)}
                        y1={cy}
                        y2={cy}
                        className="viz-line viz-line--thin"
                        style={{ stroke: g.color }}
                      />
                      <line
                        x1={cx}
                        x2={cx}
                        y1={y(Math.max(p.measured.value - p.measured.err, yDomain[0]))}
                        y2={y(p.measured.value + p.measured.err)}
                        className="viz-line viz-line--thin"
                        style={{ stroke: g.color }}
                      />
                      <Marker
                        shape={g.marker}
                        color={g.color}
                        x={cx}
                        y={cy}
                        onMouseEnter={() => onHover({ group: g, population: p })}
                        onMouseLeave={() => onHover(null)}
                      />
                    </g>
                  )
                })}
              </g>
            ))}
          </g>
        )
      }}
    </Chart>
  )
}

/** eta over ETA_RANGE of the halo or of each survey of the Milky Way, as the readout prints it. */
function etaText(galaxy: string, auriga: StatusData, milkyWay: MilkyWayData): string {
  const [lo, hi] = ETA_RANGE
  if (galaxy === MILKY_WAY) {
    const parts = milkyWay.surveys.map((survey) => {
      const range = survey.ranges.find((r) => r.lo === lo && r.hi === hi)
      return `$${(range?.eta.value ?? Number.NaN).toFixed(2)}$ with ${surveyName(survey.name)}`
    })
    return `${MILKY_WAY}: $\\eta$ of ${parts.join(', ')} over ${lo}-${hi} Gyr.`
  }
  const halo = auriga.halos.find((h) => h.label === galaxy)
  const eta = halo === undefined ? Number.NaN : (ageRange(halo, lo, hi)?.eta.value ?? Number.NaN)
  return `${galaxy}: $\\eta = ${eta.toFixed(2)}$ over ${lo}-${hi} Gyr.`
}

function Readout({
  hovered,
  selected,
  auriga,
  milkyWay
}: {
  readonly hovered: Hovered | null
  readonly selected: string | null
  readonly auriga: StatusData
  readonly milkyWay: MilkyWayData
}): ReactNode {
  if (hovered !== null) {
    const { group, population: p } = hovered
    return (
      <p className="viz-note">
        <Swatch color={group.color} />
        <MathLabel
          label={`${p.label}: $\\sigma_z = ${p.sigmaZ.value.toFixed(1)} \\pm ${p.sigmaZ.err.toFixed(1)}$ km/s, $h_Z = ${p.measured.value.toFixed(3)} \\pm ${p.measured.err.toFixed(3)}$ kpc, tracer $${p.tracer.toFixed(3)}$ kpc.`}
        />
      </p>
    )
  }
  if (selected !== null) {
    return (
      <p className="viz-note">
        <Swatch color={selected === MILKY_WAY ? MILKY_WAY_GROUP.color : haloColor(selected)} />
        <MathLabel label={etaText(selected, auriga, milkyWay)} />
      </p>
    )
  }
  return (
    <p className="viz-note">
      Point at a population to read its heights, or pick a galaxy to read its{' '}
      <MathLabel label="$\eta$" />.
    </p>
  )
}

const ENTRIES: readonly LegendEntry[] = [
  { label: 'Auriga', marker: 'circle' },
  ...Object.entries(SURVEY_MARKERS).map(([name, marker]) => ({
    label: surveyName(name),
    marker
  })),
  { label: 'tracer', dash: 'solid' },
  { label: '$h_Z \\propto \\sigma_z$', dash: 'dotted', color: 'var(--viz-muted)' },
  { label: '$h_Z \\propto \\sigma_z^2$', dash: 'dashed', color: 'var(--viz-muted)' }
]

/**
 * The measured h_Z of every mono-age population of the seven Auriga halos and of the Milky
 * Way against its sigma_z, on logarithmic axes, with the h_Z of the isothermal tracer of each
 * population joined in order of sigma_z, and the slopes eta = 1 and eta = 2 through the
 * median Auriga population.
 */
export default function ScalingChart(): ReactNode {
  const [ref, width] = useElementWidth<HTMLDivElement>()
  const revealed = useRevealed(ref)
  const auriga = useStatusData()
  const milkyWay = useRecord(MILKY_WAY_PATH, parseMilkyWay)
  const [selected, setSelected] = useState<string | null>(null)
  const [hovered, setHovered] = useState<Hovered | null>(null)
  if (auriga.value === null || milkyWay.value === null) {
    return (
      <div className="viz" ref={ref}>
        <Pending error={auriga.error ?? milkyWay.error} />
      </div>
    )
  }
  return (
    <div className="viz" ref={ref}>
      <Plot
        groups={groupsOf(auriga.value, milkyWay.value)}
        width={width}
        selected={selected}
        revealed={revealed}
        onHover={setHovered}
      />
      <Legend entries={ENTRIES} />
      <div className="viz-controls">
        <HaloPicker selected={selected} onSelect={setSelected} others={[MILKY_WAY_GROUP]} />
      </div>
      <Readout
        hovered={hovered}
        selected={selected}
        auriga={auriga.value}
        milkyWay={milkyWay.value}
      />
    </div>
  )
}
