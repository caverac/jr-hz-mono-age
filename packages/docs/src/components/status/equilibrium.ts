/**
 * The record that `experiments auriga equilibrium` writes for one halo
 * (`assets/data/auriga/<stem>-equilibrium.json`): the mean vertical motions of every
 * mono-age population of the annulus, inside the self-consistent window of its h_Z.
 */

import { type Estimate, estimate, field, list, num } from '@site/src/components/status/record'

/** The mean motions of one population; all vanish in a steady, symmetric and flat disk. */
export interface MotionBin {
  readonly ageLo: number
  readonly ageHi: number
  readonly nStars: number
  /** kpc, in the self-consistent window */
  readonly hz: Estimate
  /** <sign(z) v_z> / <|z|> (1/Gyr) */
  readonly thickening: Estimate
  /** <v_z> (km/s) */
  readonly bending: Estimate
  /** (M_+ - M_-) / (M_+ + M_-) */
  readonly asymmetry: Estimate
  /** amplitude of the m = 1 term of the mean z (kpc) */
  readonly warp: Estimate
}

/** Check and read the record of the mean motions of one halo. */
export function parseEquilibrium(record: unknown): readonly MotionBin[] {
  return list(record, 'bins').map((bin) => {
    const motions = field(bin, 'motions')
    return {
      ageLo: num(bin, 'age_lo'),
      ageHi: num(bin, 'age_hi'),
      nStars: num(bin, 'n_stars'),
      hz: estimate(bin, 'hz'),
      thickening: estimate(motions, 'thickening'),
      bending: estimate(motions, 'bending'),
      asymmetry: estimate(motions, 'asymmetry'),
      warp: estimate(motions, 'warp')
    }
  })
}
