import React, { type ReactNode } from 'react'

/** The lengths (kpc) the bar may show; it shows the one closest to `TARGET_PX` on screen. */
const ROUND_LENGTHS_KPC = [1, 2, 5, 10, 20, 50, 100] as const
const TARGET_PX = 110
const HEIGHT = 30
const TICK = 5

/**
 * Pixels per kpc at the distance of the center of the disk, for a perspective camera
 * with a vertical field of view `fieldOfViewDeg` at `distanceKpc` from the center and a
 * view `heightPx` tall: the view spans 2 d tan(fov / 2) kpc there.
 */
export function pixelsPerKpc(
  heightPx: number,
  distanceKpc: number,
  fieldOfViewDeg: number
): number {
  return heightPx / (2 * distanceKpc * Math.tan((fieldOfViewDeg * Math.PI) / 180 / 2))
}

/** The round length (kpc) whose bar is closest to `TARGET_PX` long. */
export function roundLength(pxPerKpc: number): number {
  return ROUND_LENGTHS_KPC.reduce((best, length) =>
    Math.abs(length * pxPerKpc - TARGET_PX) < Math.abs(best * pxPerKpc - TARGET_PX) ? length : best
  )
}

interface ScaleBarProps {
  readonly pxPerKpc: number
}

/**
 * A bar of a round length in kpc. With a perspective camera, a length on screen
 * depends on depth: the bar is exact at the depth of the center of the disk.
 */
export default function ScaleBar({ pxPerKpc }: ScaleBarProps): ReactNode {
  const kpc = roundLength(pxPerKpc)
  const length = kpc * pxPerKpc
  const y = HEIGHT - TICK - 1
  return (
    <svg
      className="scale-bar"
      width={length + 2}
      height={HEIGHT}
      role="img"
      aria-label={`${kpc} kpc at the center of the disk`}
    >
      <title>{`${kpc} kpc at the depth of the center of the disk`}</title>
      <text x={1 + length / 2} y={y - 6} textAnchor="middle">
        {kpc} kpc
      </text>
      <path d={`M1,${y - TICK} V${y} H${1 + length} V${y - TICK}`} />
    </svg>
  )
}
