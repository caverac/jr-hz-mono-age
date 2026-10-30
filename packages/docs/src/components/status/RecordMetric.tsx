import MathLabel from '@site/src/components/MathLabel'
import { JR_RELATION_PATH, parseJrRelation } from '@site/src/components/status/jrRelation'
import { MILKY_WAY_PATH, parseMilkyWay } from '@site/src/components/status/milkyWay'
import { MW_CHECKS_PATH, parseMwChecks } from '@site/src/components/status/mwChecks'
import { CHECK_METRICS, JR_METRICS, MW_METRICS } from '@site/src/components/status/recordMetrics'
import useRecord from '@site/src/components/status/useRecord'
import React, { type ReactNode } from 'react'

/**
 * A component that quotes the numbers *metrics* computes from the record at *path*, by
 * name, so that the prose of a page cannot drift from its record.
 */
function recordMetric<T, M extends Record<string, (data: T) => string>>(
  path: string,
  parse: (record: unknown) => T,
  metrics: M
): (props: { readonly name: keyof M & string }) => ReactNode {
  return function RecordMetricValue({ name }: { readonly name: keyof M & string }): ReactNode {
    const { value, error } = useRecord(path, parse)
    if (error !== null) {
      return <span className="metric metric--error">(no data: {error})</span>
    }
    return (
      <span className="metric">
        {value === null ? '(loading)' : <MathLabel label={metrics[name](value)} />}
      </span>
    )
  }
}

/** A number of the Milky Way page, computed from its record. */
export const MwMetric = recordMetric(MILKY_WAY_PATH, parseMilkyWay, MW_METRICS)

/** A number of the page across galaxies, computed from its record. */
export const JrMetric = recordMetric(JR_RELATION_PATH, parseJrRelation, JR_METRICS)

/** A number of the checks of the Milky Way test, computed from their record. */
export const CheckMetric = recordMetric(MW_CHECKS_PATH, parseMwChecks, CHECK_METRICS)
