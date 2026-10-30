import MathLabel from '@site/src/components/MathLabel'
import { type MetricName, STATUS_METRICS } from '@site/src/components/status/statusMetrics'
import useStatusData from '@site/src/components/status/useStatusData'
import React, { type ReactNode } from 'react'

export type { MetricName } from '@site/src/components/status/statusMetrics'

/**
 * A number of the status pages, computed from the records when the page loads, so that
 * the prose cannot drift from the data.
 */
export default function Metric({ name }: { readonly name: MetricName }): ReactNode {
  const { value, error } = useStatusData()
  if (error !== null) {
    return <span className="metric metric--error">(no data: {error})</span>
  }
  return (
    <span className="metric">
      {value === null ? '(loading)' : <MathLabel label={STATUS_METRICS[name](value)} />}
    </span>
  )
}
