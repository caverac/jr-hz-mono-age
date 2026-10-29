import MathLabel from '@site/src/components/MathLabel'
import { haloColor } from '@site/src/components/render/halos'
import { Chips, fadeClass, HaloPicker, Pending } from '@site/src/components/status/controls'
import type { AgeBin, StatusData } from '@site/src/components/status/data'
import { PREDICTOR_LABELS, PREDICTORS } from '@site/src/components/status/HeightRatioChart'
import {
  median,
  type Predictor,
  populations,
  ratio,
  ratios,
  within
} from '@site/src/components/status/metrics'
import useStatusData from '@site/src/components/status/useStatusData'
import Chart, { Axis, bandPath, linePath, ticksWithin } from '@site/src/components/viz/Chart'
import { useElementWidth, useRevealed } from '@site/src/components/viz/hooks'
import { Legend, type LegendEntry, Marker, Swatch } from '@site/src/components/viz/marks'
import * as d3 from 'd3'
import React, { type ReactNode, useState } from 'react'

const HEIGHT = 440
const HEIGHT_TICKS = [0.2, 0.3, 0.5, 0.7, 1, 1.5, 2, 3, 5]
/** The ticks of an x axis narrower than NARROW px, where the full set would collide. */
const SPARSE_TICKS = [0.2, 0.5, 1, 2, 5]
const NARROW = 380
/** The band of relative agreement drawn around the line of equality. */
const TOLERANCE = 0.1

type Population = AgeBin & { readonly halo: string }

/** The h_Z (kpc) that `predictor` gives the population, as its ratio times the measured h_Z. */
function predicted(bin: AgeBin, predictor: Predictor): number {
  return ratio(bin, predictor) * bin.measured.value
}

interface Hovered {
  readonly population: Population
}

interface PlotProps {
  readonly data: StatusData
  readonly predictor: Predictor
  readonly width: number
  readonly selected: string | null
  readonly revealed: boolean
  readonly onHover: (hovered: Hovered | null) => void
}

function Plot({ data, predictor, width, selected, revealed, onHover }: PlotProps): ReactNode {
  const shown = populations(data).filter(
    (p) => Number.isFinite(p.measured.value) && Number.isFinite(predicted(p, predictor))
  )
  // One domain for every prediction, so that the axes stay put while the markers move.
  const values = populations(data).flatMap((p) => [
    p.measured.value,
    ...PREDICTORS.map((q) => predicted(p, q))
  ])
  const finite = values.filter((v) => Number.isFinite(v) && v > 0)
  const domain: readonly [number, number] = [
    (d3.min(finite) ?? 0.2) / 1.1,
    (d3.max(finite) ?? 5) * 1.1
  ]
  return (
    <Chart width={width} height={HEIGHT} xLabel="$h_Z$ (kpc)" yLabel="$h_\mathrm{pred}$ (kpc)">
      {(inner) => {
        const x = d3.scaleLog().domain(domain).range([0, inner.width])
        const y = d3.scaleLog().domain(domain).range([inner.height, 0])
        const edge = (factor: number) => domain.map((h) => [x(h), y(h * factor)] as const)
        return (
          <g className={revealed ? 'viz-reveal viz-reveal--in' : 'viz-reveal'}>
            <g clipPath={inner.clip}>
              <path d={bandPath(edge(1 - TOLERANCE), edge(1 + TOLERANCE))} className="viz-band" />
            </g>
            <Axis
              scale={y}
              orientation="left"
              tickValues={ticksWithin(HEIGHT_TICKS, domain)}
              tickFormat={(v) => String(v)}
              gridLength={inner.width}
            />
            <Axis
              scale={x}
              orientation="bottom"
              transform={`translate(0,${inner.height})`}
              tickValues={ticksWithin(inner.width < NARROW ? SPARSE_TICKS : HEIGHT_TICKS, domain)}
              tickFormat={(v) => String(v)}
              gridLength={inner.height}
            />
            <path
              d={linePath(edge(1))}
              pathLength={1}
              className="viz-line viz-line--thin viz-draw"
            />
            {shown.map((p, i) => {
              const color = haloColor(p.halo)
              const px = x(p.measured.value)
              const py = y(predicted(p, predictor))
              return (
                <g key={`${p.halo}-${p.ageLo}`} className={fadeClass(selected, p.halo)}>
                  <g className="viz-move" style={{ transform: `translate(${px}px, ${py}px)` }}>
                    <g className="viz-pop" style={{ animationDelay: `${300 + i * 14}ms` }}>
                      <line
                        x1={x(Math.max(p.measured.value - p.measured.err, domain[0])) - px}
                        x2={x(p.measured.value + p.measured.err) - px}
                        y1={0}
                        y2={0}
                        className="viz-line viz-line--thin"
                        style={{ stroke: color }}
                      />
                      <Marker
                        shape="circle"
                        color={color}
                        x={0}
                        y={0}
                        onMouseEnter={() => onHover({ population: p })}
                        onMouseLeave={() => onHover(null)}
                      />
                    </g>
                  </g>
                </g>
              )
            })}
          </g>
        )
      }}
    </Chart>
  )
}

function Readout({
  data,
  predictor,
  hovered
}: {
  readonly data: StatusData
  readonly predictor: Predictor
  readonly hovered: Hovered | null
}): ReactNode {
  if (hovered !== null) {
    const p = hovered.population
    const r = p.ratios[predictor]
    return (
      <p className="viz-note">
        <Swatch color={haloColor(p.halo)} />
        <MathLabel
          label={`${p.halo}, ${p.ageLo}-${p.ageHi} Gyr: $h_Z = ${p.measured.value.toFixed(3)} \\pm ${p.measured.err.toFixed(3)}$ kpc, ${PREDICTOR_LABELS[predictor]} $${predicted(p, predictor).toFixed(3)}$ kpc, a ratio of $${r.value.toFixed(2)} \\pm ${r.err.toFixed(2)}$.`}
        />
      </p>
    )
  }
  const all = ratios(data, predictor)
  return (
    <p className="viz-note">
      <MathLabel
        label={`${PREDICTOR_LABELS[predictor]}: ${within(data, predictor, TOLERANCE)} of ${all.length} populations within ${100 * TOLERANCE} percent, with a median ratio of $${median(all).toFixed(2)}$. Point at a population to read its heights.`}
      />
    </p>
  )
}

const ENTRIES: readonly LegendEntry[] = [
  { label: 'population', marker: 'circle' },
  { label: '$h_\\mathrm{pred} = h_Z$', dash: 'solid' },
  { label: `$|h_\\mathrm{pred}/h_Z - 1| \\le ${TOLERANCE}$`, band: true }
]

/**
 * The h_Z that one prediction gives every mono-age population of the seven Auriga halos
 * against the h_Z measured in the self-consistent window, on logarithmic axes. The markers
 * move to their new places when the reader picks another prediction.
 */
export default function PredictionChart(): ReactNode {
  const [ref, width] = useElementWidth<HTMLDivElement>()
  const revealed = useRevealed(ref)
  const { value, error } = useStatusData()
  const [predictor, setPredictor] = useState<Predictor>('tracer')
  const [selected, setSelected] = useState<string | null>(null)
  const [hovered, setHovered] = useState<Hovered | null>(null)
  if (value === null) {
    return (
      <div className="viz" ref={ref}>
        <Pending error={error} />
      </div>
    )
  }
  return (
    <div className="viz" ref={ref}>
      <Plot
        data={value}
        predictor={predictor}
        width={width}
        selected={selected}
        revealed={revealed}
        onHover={setHovered}
      />
      <Legend entries={ENTRIES} />
      <div className="viz-controls">
        <Chips
          label="Prediction"
          options={PREDICTORS.map((p) => [p, PREDICTOR_LABELS[p]] as const)}
          isPressed={(p) => p === predictor}
          onToggle={setPredictor}
        />
        <HaloPicker selected={selected} onSelect={setSelected} />
      </div>
      <Readout data={value} predictor={predictor} hovered={hovered} />
    </div>
  )
}
