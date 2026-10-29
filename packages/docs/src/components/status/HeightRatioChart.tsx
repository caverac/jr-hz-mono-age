import MathLabel from '@site/src/components/MathLabel'
import { haloColor } from '@site/src/components/render/halos'
import {
  Chips,
  fadeClass,
  HaloPicker,
  Pending,
  toggled
} from '@site/src/components/status/controls'
import type { AgeBin, StatusData } from '@site/src/components/status/data'
import { type Predictor, ratio } from '@site/src/components/status/metrics'
import useStatusData from '@site/src/components/status/useStatusData'
import Chart, { Axis, linePath, ticksWithin } from '@site/src/components/viz/Chart'
import { useElementWidth } from '@site/src/components/viz/hooks'
import { dashClass, Legend, Marker, Swatch, type LegendEntry } from '@site/src/components/viz/marks'
import * as d3 from 'd3'
import React, { type ReactNode, useState } from 'react'

const HEIGHT = 360
const AGE_DOMAIN: readonly [number, number] = [0, 10]
const RATIO_TICKS = [0.3, 0.5, 0.7, 1, 1.5, 2, 3, 5]
/** The band of relative agreement drawn around a ratio of 1. */
const TOLERANCE = 0.1

/** How each prediction is drawn: line, marker and fill. */
const STYLE = {
  tracer: { dash: 'solid', marker: 'circle', open: false },
  tracerSnapshot: { dash: 'dotted', marker: 'diamond', open: false },
  jeansSigma: { dash: 'dashed', marker: 'triangle', open: false },
  jeans: { dash: 'dotted', marker: 'triangle', open: true },
  sheet: { dash: 'dashed', marker: 'circle', open: true }
} as const

const LABELS: Record<Predictor, string> = {
  tracer: 'tracer, fitted $\\Phi$',
  tracerSnapshot: 'tracer, snapshot $\\Phi$',
  jeansSigma: 'Jeans, $\\sigma_z(z)$',
  jeans: 'Jeans, $\\sigma_z(z)$ and tilt',
  sheet: 'sheet'
}

const PREDICTORS: readonly Predictor[] = [
  'tracer',
  'tracerSnapshot',
  'jeansSigma',
  'jeans',
  'sheet'
]

interface Hovered {
  readonly halo: string
  readonly bin: AgeBin
}

function format(value: number): string {
  return Number.isFinite(value) ? value.toFixed(2) : 'none'
}

/** The readout of the hovered population, or how to get one. */
function Readout({ hovered }: { readonly hovered: Hovered | null }): ReactNode {
  if (hovered === null) {
    return <p className="viz-note">Point at a population to read its ratios.</p>
  }
  const { halo, bin } = hovered
  const parts = PREDICTORS.map(
    (p) => `${LABELS[p]}: $${format(ratio(bin, p))} \\pm ${format(bin.ratios[p].err)}$`
  )
  return (
    <p className="viz-note">
      <Swatch color={haloColor(halo)} />
      <MathLabel
        label={`${halo}, ${bin.ageLo}-${bin.ageHi} Gyr, $h_Z = ${bin.measured.value.toFixed(3)} \\pm ${bin.measured.err.toFixed(3)}$ kpc. ${parts.join('; ')}.`}
      />
    </p>
  )
}

interface PlotProps {
  readonly data: StatusData
  readonly width: number
  readonly shown: ReadonlySet<Predictor>
  readonly selected: string | null
  readonly onHover: (hovered: Hovered | null) => void
}

function Plot({ data, width, shown, selected, onHover }: PlotProps): ReactNode {
  const values = data.halos.flatMap((halo) =>
    halo.bins.flatMap((bin) => PREDICTORS.map((p) => ratio(bin, p)))
  )
  const finite = values.filter(Number.isFinite)
  const domain: readonly [number, number] = [
    Math.min(0.8, d3.min(finite) ?? 1) / 1.15,
    Math.max(1.25, d3.max(finite) ?? 1) * 1.15
  ]
  return (
    <Chart width={width} height={HEIGHT} xLabel="$\tau$ (Gyr)" yLabel="$h_\mathrm{pred}/h_Z$">
      {(inner) => {
        const x = d3.scaleLinear().domain(AGE_DOMAIN).range([0, inner.width])
        const y = d3.scaleLog().domain(domain).range([inner.height, 0])
        return (
          <>
            <rect
              x={0}
              width={inner.width}
              y={y(1 + TOLERANCE)}
              height={y(1 - TOLERANCE) - y(1 + TOLERANCE)}
              className="viz-band"
            />
            <Axis
              scale={y}
              orientation="left"
              tickValues={ticksWithin(RATIO_TICKS, domain)}
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
            <line x1={0} x2={inner.width} y1={y(1)} y2={y(1)} className="viz-line viz-line--thin" />
            {data.halos.map((halo) =>
              PREDICTORS.filter((p) => shown.has(p)).map((p) => {
                const style = STYLE[p]
                const points = halo.bins
                  .map((bin) => ({ bin, r: ratio(bin, p) }))
                  .filter(({ r }) => Number.isFinite(r))
                const color = haloColor(halo.label)
                return (
                  <g key={`${halo.label}-${p}`} className={fadeClass(selected, halo.label)}>
                    <path
                      d={linePath(points.map(({ bin, r }) => [x(bin.ageMedian), y(r)] as const))}
                      className={`${dashClass(style.dash)} viz-line--series`}
                      style={{ stroke: color }}
                    />
                    {points.map(({ bin, r }) => {
                      const spread = bin.ratios[p].err
                      return (
                        <g key={bin.ageLo}>
                          <line
                            x1={x(bin.ageMedian)}
                            x2={x(bin.ageMedian)}
                            y1={y(Math.max(r - spread, domain[0]))}
                            y2={y(r + spread)}
                            className="viz-line viz-line--thin"
                            style={{ stroke: color }}
                          />
                          <Marker
                            shape={style.marker}
                            open={style.open}
                            color={color}
                            x={x(bin.ageMedian)}
                            y={y(r)}
                            onMouseEnter={() => onHover({ halo: halo.label, bin })}
                            onMouseLeave={() => onHover(null)}
                          />
                        </g>
                      )
                    })}
                  </g>
                )
              })
            )}
          </>
        )
      }}
    </Chart>
  )
}

/**
 * The scale height that the sheet and the isothermal tracer predict for every mono-age
 * population, over the scale height measured in the self-consistent window, against the
 * median age of the population.
 */
export default function HeightRatioChart(): ReactNode {
  const [ref, width] = useElementWidth<HTMLDivElement>()
  const { value, error } = useStatusData()
  const [shown, setShown] = useState<ReadonlySet<Predictor>>(new Set(['tracer', 'sheet']))
  const [selected, setSelected] = useState<string | null>(null)
  const [hovered, setHovered] = useState<Hovered | null>(null)
  const entries: readonly LegendEntry[] = [
    ...PREDICTORS.filter((p) => shown.has(p)).map((p) => ({ label: LABELS[p], ...STYLE[p] })),
    { label: `$|h_\\mathrm{pred}/h_Z - 1| \\le ${TOLERANCE}$`, band: true }
  ]
  return (
    <div className="viz" ref={ref}>
      {value === null ? (
        <Pending error={error} />
      ) : (
        <>
          <Plot data={value} width={width} shown={shown} selected={selected} onHover={setHovered} />
          <Legend entries={entries} />
          <div className="viz-controls">
            <Chips
              label="Predictions"
              options={PREDICTORS.map((p) => [p, LABELS[p]] as const)}
              isPressed={(p) => shown.has(p)}
              onToggle={(p) => setShown((current) => toggled(current, p))}
            />
            <HaloPicker selected={selected} onSelect={setSelected} />
          </div>
          <Readout hovered={hovered} />
        </>
      )}
    </div>
  )
}
