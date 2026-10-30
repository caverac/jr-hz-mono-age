/**
 * The record that `experiments auriga jr-relation` writes (`assets/data/auriga/jr-relation.json`):
 * the proxy J_R,w of each halo against its whole-disk h_Z, in the windows of Palicio et al.
 * (2025) and in the self-consistent one, measured and predicted as a sum of isothermal
 * tracers, with the line through each, the same lines without each halo, and the line of
 * Palicio et al. (2025, Fig. 9).
 */

import { field, list, num } from '@site/src/components/status/record'

export const JR_RELATION_PATH = '/data/auriga/jr-relation.json'

export interface JrPoint {
  readonly halo: string
  /** kpc km/s */
  readonly jrW: number
  /** kpc */
  readonly hzPalicio: number
  readonly hzScaled: number
  /** kpc, predicted by the sum of tracers of every ring and age bin */
  readonly hzPalicioMixture: number
  readonly hzScaledMixture: number
}

/** The least-squares line J_R,w = intercept + slope h_Z, slope in km/s. */
export interface Line {
  readonly slope: number
  readonly slopeErr: number
  readonly intercept: number
  readonly interceptErr: number
}

/** The lines of J_R,w against the measured and the predicted (`Mixture`) h_Z, in both windows. */
export interface Lines {
  readonly linePalicio: Line
  readonly lineScaled: Line
  readonly linePalicioMixture: Line
  readonly lineScaledMixture: Line
}

/** The lines refitted without one halo. */
export interface LeftOut {
  readonly halo: string
  readonly lines: Lines
}

export interface JrRelationData extends Lines {
  readonly points: readonly JrPoint[]
  readonly leaveOneOut: readonly LeftOut[]
  /** The line of Palicio et al. (2025, Fig. 9): slope in km/s, intercept in kpc km/s. */
  readonly palicioSlope: number
  readonly palicioIntercept: number
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

function lines(value: unknown): Lines {
  return {
    linePalicio: line(value, 'line_palicio'),
    lineScaled: line(value, 'line_scaled'),
    linePalicioMixture: line(value, 'line_palicio_mixture'),
    lineScaledMixture: line(value, 'line_scaled_mixture')
  }
}

/** Check and read the record of the relation across the halos. */
export function parseJrRelation(record: unknown): JrRelationData {
  return {
    points: list(record, 'points').map((point) => ({
      halo: `Au${num(point, 'halo')}`,
      jrW: num(point, 'jr_w'),
      hzPalicio: num(point, 'hz_palicio'),
      hzScaled: num(point, 'hz_scaled'),
      hzPalicioMixture: num(point, 'hz_palicio_mixture'),
      hzScaledMixture: num(point, 'hz_scaled_mixture')
    })),
    ...lines(field(record, 'lines')),
    leaveOneOut: list(record, 'leave_one_out').map((left) => ({
      halo: `Au${num(left, 'halo')}`,
      lines: lines(field(left, 'lines'))
    })),
    palicioSlope: num(record, 'palicio_slope'),
    palicioIntercept: num(record, 'palicio_intercept')
  }
}
