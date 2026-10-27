import MathLabel from '@site/src/components/MathLabel'
import { HALOS } from '@site/src/components/render/halos'
import React, { type ReactNode } from 'react'

/** What a chart shows while the records load, or when they cannot be read. */
export function Pending({ error }: { readonly error: string | null }): ReactNode {
  return (
    <p className={error === null ? 'viz-note' : 'viz-note viz-note--error'}>
      {error === null ? 'Loading the records.' : `The records could not be read: ${error}`}
    </p>
  )
}

interface ChipsProps<T extends string> {
  /** The choices and their labels, with inline math between dollar signs. */
  readonly options: readonly (readonly [T, string])[]
  readonly isPressed: (option: T) => boolean
  readonly onToggle: (option: T) => void
  readonly label: string
}

/** A row of toggle buttons. */
export function Chips<T extends string>({
  options,
  isPressed,
  onToggle,
  label
}: ChipsProps<T>): ReactNode {
  return (
    <div className="viz-chips" role="group" aria-label={label}>
      {options.map(([option, text]) => (
        <button
          key={option}
          type="button"
          className="viz-chip"
          aria-pressed={isPressed(option)}
          onClick={() => onToggle(option)}
        >
          <MathLabel label={text} />
        </button>
      ))}
    </div>
  )
}

const ALL = 'all'

interface HaloPickerProps {
  /** The highlighted halo, or null for all. */
  readonly selected: string | null
  readonly onSelect: (halo: string | null) => void
}

/** Buttons that highlight one halo in a chart, or none. */
export function HaloPicker({ selected, onSelect }: HaloPickerProps): ReactNode {
  const options: readonly (readonly [string, string])[] = [
    [ALL, 'all'],
    ...HALOS.map((halo) => [halo.label, halo.label] as const)
  ]
  return (
    <Chips
      label="Highlight a halo"
      options={options}
      isPressed={(option) => (option === ALL ? selected === null : option === selected)}
      onToggle={(option) => onSelect(option === ALL || option === selected ? null : option)}
    />
  )
}

/** The class that fades the marks of `halo` when another halo is highlighted. */
export function fadeClass(selected: string | null, halo: string): string {
  return selected !== null && selected !== halo ? 'viz-faded' : ''
}

/** `set` with `item` added if it was absent, or removed if it was present. */
export function toggled<T>(set: ReadonlySet<T>, item: T): ReadonlySet<T> {
  const next = new Set(set)
  if (next.has(item)) {
    next.delete(item)
  } else {
    next.add(item)
  }
  return next
}
