import MathLabel from '@site/src/components/MathLabel'
import React, { type ReactNode } from 'react'

/** The positions of a slider: enough that a logarithmic one moves smoothly. */
const STEPS = 400

interface SliderProps {
  /** The name of the quantity, with inline math between dollar signs. */
  readonly label: string
  readonly value: number
  readonly min: number
  readonly max: number
  /** Whether the slider moves in equal ratios rather than in equal steps. */
  readonly log?: boolean
  /** The value as the readout shows it, with inline math between dollar signs. */
  readonly format: (value: number) => string
  readonly onChange: (value: number) => void
}

/** A labeled range input with its value beside it, linear or logarithmic. */
export default function Slider({
  label,
  value,
  min,
  max,
  log = false,
  format,
  onChange
}: SliderProps): ReactNode {
  const toPosition = (v: number): number =>
    log ? (STEPS * Math.log(v / min)) / Math.log(max / min) : (STEPS * (v - min)) / (max - min)
  const fromPosition = (p: number): number =>
    log ? min * (max / min) ** (p / STEPS) : min + ((max - min) * p) / STEPS
  return (
    <label className="viz-slider">
      <span className="viz-slider__label">
        <MathLabel label={label} />
      </span>
      <input
        type="range"
        min={0}
        max={STEPS}
        step={1}
        value={Math.round(toPosition(value))}
        onChange={(event) => onChange(fromPosition(Number(event.target.value)))}
      />
      <span className="viz-slider__value">
        <MathLabel label={format(value)} />
      </span>
    </label>
  )
}
