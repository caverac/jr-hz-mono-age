/**
 * The record that `experiments auriga tracer-sum` writes for one halo
 * (`assets/data/auriga/<stem>-tracer-sum.json`): the whole-disk h_Z of the star particles
 * and its predictions as a sum of isothermal tracers, one per ring (`single`) or one per
 * ring and age bin (`mixture`), in Palicio's window and in the self-consistent one.
 */

import { type Estimate, estimate, field, list, num } from '@site/src/components/status/record'

/** h_Z (kpc), or a ratio of two, in Palicio's window and in the self-consistent one. */
export interface SumHeights {
  readonly palicio: Estimate
  readonly scaled: Estimate
}

/** One radial range: the mass each prediction leaves out, the heights and their ratios. */
export interface SumRange {
  readonly rLo: number
  readonly rHi: number
  readonly unmodeledSingle: number
  readonly unmodeledMixture: number
  readonly measured: SumHeights
  readonly single: SumHeights
  readonly mixture: SumHeights
  readonly ratioSingle: SumHeights
  readonly ratioMixture: SumHeights
}

export interface TracerSum {
  readonly rings: readonly SumRange[]
  readonly whole: SumRange
}

function heights(value: unknown, key: string): SumHeights {
  const item = field(value, key)
  return { palicio: estimate(item, 'palicio'), scaled: estimate(item, 'scaled') }
}

function parseRange(range: unknown): SumRange {
  const unmodeled = field(range, 'unmodeled')
  return {
    rLo: num(range, 'r_lo'),
    rHi: num(range, 'r_hi'),
    unmodeledSingle: num(unmodeled, 'single'),
    unmodeledMixture: num(unmodeled, 'mixture'),
    measured: heights(range, 'measured'),
    single: heights(range, 'single'),
    mixture: heights(range, 'mixture'),
    ratioSingle: heights(range, 'ratio_single'),
    ratioMixture: heights(range, 'ratio_mixture')
  }
}

/** Check and read the record of the tracer sum of one halo. */
export function parseTracerSum(record: unknown): TracerSum {
  const sum = field(record, 'tracer_sum')
  return { rings: list(sum, 'rings').map(parseRange), whole: parseRange(field(sum, 'whole')) }
}
