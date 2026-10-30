/**
 * The record of `experiments milky-way checks`: the Milky Way test remeasured with the ages
 * of every star drawn again from their errors, and with the bright and the faint half of the
 * age tracers of each survey.
 */

import { type MwSurvey, parseSurvey } from '@site/src/components/status/milkyWay'
import { field, list, num, RecordError, str } from '@site/src/components/status/record'

export const MW_CHECKS_PATH = '/data/milky-way/checks.json'

/** One survey measured with the tracers brighter and fainter than G = `gSplit`. */
export interface MagnitudeSplit {
  readonly survey: string
  readonly gSplit: number
  readonly bright: MwSurvey
  readonly faint: MwSurvey
}

/** The surveys with the ages drawn again, one list per realization, and split by magnitude. */
export interface MwChecks {
  readonly nBootstrap: number
  readonly ageErrors: readonly (readonly MwSurvey[])[]
  readonly magnitude: readonly MagnitudeSplit[]
}

/** Check and read the record of the Milky Way checks. */
export function parseMwChecks(record: unknown): MwChecks {
  return {
    nBootstrap: num(record, 'n_bootstrap'),
    ageErrors: list(record, 'age_errors').map((realization) => {
      if (!Array.isArray(realization)) {
        throw new RecordError('a realization of "age_errors" is not a list of surveys')
      }
      return realization.map(parseSurvey)
    }),
    magnitude: list(record, 'magnitude').map((split) => ({
      survey: str(split, 'survey'),
      gSplit: num(split, 'g_split'),
      bright: parseSurvey(field(split, 'bright')),
      faint: parseSurvey(field(split, 'faint'))
    }))
  }
}
