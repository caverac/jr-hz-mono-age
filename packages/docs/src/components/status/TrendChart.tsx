import HaloRowsChart, {
  type RowPanel,
  type RowSeries
} from '@site/src/components/status/HaloRowsChart'
import { AGE_RANGES, exponentComparisons } from '@site/src/components/status/metrics'
import React, { type ReactNode } from 'react'

const SERIES: readonly RowSeries[] = [
  { key: 'tracer', label: 'tracer', marker: 'square', open: true },
  { key: 'sheet', label: 'sheet', marker: 'triangle', open: true }
]

/**
 * The trend d ln(h_pred / h_Z) / d ln age of the tracer and of the sheet in every halo, over
 * the two ranges of ages. A prediction that reproduces the age dependence of h_Z lies on 0.
 */
export default function TrendChart(): ReactNode {
  return (
    <HaloRowsChart
      series={SERIES}
      xLabel="$d\ln(h_\mathrm{pred}/h_Z)/d\ln\tau$"
      build={(data): readonly RowPanel[] =>
        AGE_RANGES.map(([lo, hi]) => ({
          label: `${lo}-${hi} Gyr`,
          rows: exponentComparisons(data)
            .filter((c) => c.lo === lo && c.hi === hi)
            .map((c) => ({
              halo: c.halo,
              values: { tracer: c.tracerTrend, sheet: c.sheetTrend }
            }))
        }))
      }
    />
  )
}
