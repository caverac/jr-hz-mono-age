/**
 * The record that `experiments auriga jr-relation` writes (`assets/data/auriga/jr-relation.json`):
 * the proxy J_R,w of each halo against its whole-disk h_Z, in the windows of Palicio et al.
 * (2025) and in the self-consistent one, with the line through each.
 */

import { field, list, num } from '@site/src/components/status/data'

export const JR_RELATION_PATH = '/data/auriga/jr-relation.json'

export interface JrPoint {
  readonly halo: string
  /** kpc km/s */
  readonly jrW: number
  /** kpc */
  readonly hzPalicio: number
  readonly hzScaled: number
}

/** The least-squares line J_R,w = intercept + slope h_Z, slope in km/s. */
export interface Line {
  readonly slope: number
  readonly slopeErr: number
  readonly intercept: number
  readonly interceptErr: number
}

export interface JrRelationData {
  readonly points: readonly JrPoint[]
  readonly linePalicio: Line
  readonly lineScaled: Line
  /** The slope of Palicio et al. (2025) in km/s. */
  readonly palicioSlope: number
}

function line(value: unknown, key: string): Line {
  const item = field(value, key)
  return {
    slope: num(item, 'slope'),
    slopeErr: num(item, 'slope_err'),
    intercept: num(item, 'intercept'),
    interceptErr: num(item, 'intercept_err')
  }
}

/** Check and read the record of the relation across the halos. */
export function parseJrRelation(record: unknown): JrRelationData {
  return {
    points: list(record, 'points').map((point) => ({
      halo: `Au${num(point, 'halo')}`,
      jrW: num(point, 'jr_w'),
      hzPalicio: num(point, 'hz_palicio'),
      hzScaled: num(point, 'hz_scaled')
    })),
    linePalicio: line(record, 'line_palicio'),
    lineScaled: line(record, 'line_scaled'),
    palicioSlope: num(record, 'palicio_slope')
  }
}
