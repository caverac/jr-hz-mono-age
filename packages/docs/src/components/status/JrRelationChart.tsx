import { haloColor } from '@site/src/components/render/halos'
import { fadeClass, HaloPicker, Pending } from '@site/src/components/status/controls'
import {
  JR_RELATION_PATH,
  type JrRelationData,
  type Line,
  parseJrRelation
} from '@site/src/components/status/jrRelation'
import useRecord from '@site/src/components/status/useRecord'
import Chart, { Axis } from '@site/src/components/viz/Chart'
import { useElementWidth } from '@site/src/components/viz/hooks'
import { dashClass, Legend, Marker } from '@site/src/components/viz/marks'
import * as d3 from 'd3'
import React, { type ReactNode, useState } from 'react'

const HEIGHT = 360

/** The line `line` across the x domain, in pixels. */
function LinePath({
  line,
  x,
  y,
  dash
}: {
  readonly line: Line
  readonly x: d3.ScaleLinear<number, number>
  readonly y: d3.ScaleLinear<number, number>
  readonly dash: 'solid' | 'dashed' | 'dotted'
}): ReactNode {
  const [lo, hi] = x.domain()
  return (
    <line
      x1={x(lo)}
      x2={x(hi)}
      y1={y(line.intercept + line.slope * lo)}
      y2={y(line.intercept + line.slope * hi)}
      className={dashClass(dash)}
    />
  )
}

function Plot({
  data,
  width,
  selected
}: {
  readonly data: JrRelationData
  readonly width: number
  readonly selected: string | null
}): ReactNode {
  const hz = data.points.flatMap((p) => [p.hzPalicio, p.hzScaled])
  const jr = data.points.map((p) => p.jrW)
  const xDomain: readonly [number, number] = [0, (d3.max(hz) ?? 2) * 1.15]
  const yDomain: readonly [number, number] = [0, (d3.max(jr) ?? 100) * 1.2]
  const reference: Line = {
    slope: data.palicioSlope,
    slopeErr: 0,
    intercept: data.palicioIntercept,
    interceptErr: 0
  }
  return (
    <Chart width={width} height={HEIGHT} xLabel="$h_Z$ (kpc)" yLabel="$J_{R,w}$ (kpc km s$^{-1}$)">
      {(inner) => {
        const x = d3.scaleLinear().domain(xDomain).range([0, inner.width])
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
              <LinePath line={data.linePalicio} x={x} y={y} dash="solid" />
              <LinePath line={data.lineScaled} x={x} y={y} dash="dashed" />
              <LinePath line={reference} x={x} y={y} dash="dotted" />
            </g>
            {data.points.map((p) => {
              const color = haloColor(p.halo)
              return (
                <g key={p.halo} className={fadeClass(selected, p.halo)}>
                  <line
                    x1={x(p.hzPalicio)}
                    x2={x(p.hzScaled)}
                    y1={y(p.jrW)}
                    y2={y(p.jrW)}
                    className="viz-line viz-line--thin"
                    style={{ stroke: color }}
                  />
                  <Marker shape="circle" color={color} x={x(p.hzPalicio)} y={y(p.jrW)} />
                  <Marker shape="circle" open color={color} x={x(p.hzScaled)} y={y(p.jrW)} />
                </g>
              )
            })}
          </>
        )
      }}
    </Chart>
  )
}

/**
 * The proxy J_R,w of every halo against its whole-disk h_Z, in the windows of Palicio et al.
 * (2025) and in the self-consistent one, with the least-squares line through each and the
 * line of Palicio et al. (2025, Fig. 9).
 */
export default function JrRelationChart(): ReactNode {
  const [ref, width] = useElementWidth<HTMLDivElement>()
  const { value, error } = useRecord(JR_RELATION_PATH, parseJrRelation)
  const [selected, setSelected] = useState<string | null>(null)
  return (
    <div className="viz" ref={ref}>
      {value === null ? (
        <Pending error={error} />
      ) : (
        <>
          <Plot data={value} width={width} selected={selected} />
          <Legend
            entries={[
              { label: '$h_Z$ in the windows of Palicio et al.', marker: 'circle', dash: 'solid' },
              {
                label: '$h_Z$ in the self-consistent window',
                marker: 'circle',
                open: true,
                dash: 'dashed'
              },
              { label: 'Palicio et al. (2025)', dash: 'dotted' }
            ]}
          />
          <div className="viz-controls">
            <HaloPicker selected={selected} onSelect={setSelected} />
          </div>
        </>
      )}
    </div>
  )
}
