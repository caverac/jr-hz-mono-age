/** One Auriga halo at Level 4, z = 0, as the Python package names its files. */
export interface Halo {
  readonly label: string
  readonly stem: string
}

/** The seven halos of Palicio et al. (2025, Table 1). */
export const HALOS: readonly Halo[] = [2, 3, 16, 19, 23, 24, 27].map((halo) => ({
  label: `Au${halo}`,
  stem: `au${halo}-level4-Original-snap127`
}))

/**
 * The color of the halo `label` in the charts of the docs, as the CSS custom property that
 * holds its light and dark steps (`src/css/viz.css`).
 */
export function haloColor(label: string): string {
  return `var(--viz-halo-${label.toLowerCase()})`
}

/** The halo shown first. */
export const DEFAULT_HALO = 'Au24'

/** Path of the portrait of `halo` under the static directories (`experiments auriga render`). */
export function portraitPath(halo: Halo): string {
  return `/figures/${halo.stem}-portrait.jpg`
}

/** Path of the star sample of `halo` without its extension (`experiments auriga render-data`). */
export function samplePath(halo: Halo): string {
  return `/data/auriga/${halo.stem}-stars`
}
