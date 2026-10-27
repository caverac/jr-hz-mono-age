import type { Dash } from '@site/src/components/viz/marks'

/** The injected families, as `experiments window-injection` names them, with their labels and dashes. */
export const FAMILIES: readonly (readonly [string, string, Dash])[] = [
  ['exponential', 'exponential', 'solid'],
  ['sech2', '$\\mathrm{sech}^2$', 'dashed'],
  ['mixture', 'mixture', 'dotted']
]
