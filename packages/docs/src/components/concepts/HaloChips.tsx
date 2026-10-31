import { DEFAULT_HALO, type Halo, HALOS, haloColor } from '@site/src/components/render/halos'
import { Chips } from '@site/src/components/status/controls'
import React, { type ReactNode } from 'react'

/** The halo of `label`, or the default one. */
export function haloOf(label: string): Halo {
  return HALOS.find((halo) => halo.label === label) ?? HALOS[0]
}

/** Buttons that choose the one halo a chart shows. */
export default function HaloChips({
  selected,
  onSelect
}: {
  readonly selected: string
  readonly onSelect: (label: string) => void
}): ReactNode {
  return (
    <Chips
      label="Halo"
      options={HALOS.map((halo) => [halo.label, halo.label] as const)}
      isPressed={(option) => option === selected}
      onToggle={onSelect}
      swatch={haloColor}
    />
  )
}

export { DEFAULT_HALO }
