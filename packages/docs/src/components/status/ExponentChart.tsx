import HaloRowsChart, {
  type RowPanel,
  type RowSeries
} from '@site/src/components/status/HaloRowsChart'
import { AGE_RANGES, exponentComparisons } from '@site/src/components/status/metrics'
import React, { type ReactNode } from 'react'

const SERIES: readonly RowSeries[] = [
  { key: 'measured', label: 'measured', marker: 'circle', open: false },
  { key: 'tracer', label: 'tracer', marker: 'square', open: true },
  { key: 'sheet', label: 'sheet', marker: 'triangle', open: true }
]

/**
 * The age exponent beta_a of the slope a = J_R,med / h_Z in every halo, measured and with
 * the predicted h_Z of the tracer and of the sheet, over the two ranges of ages.
 */
export default function ExponentChart(): ReactNode {
  return (
    <HaloRowsChart
      series={SERIES}
      xLabel="$\beta_a$"
      build={(data): readonly RowPanel[] =>
        AGE_RANGES.map(([lo, hi]) => ({
          label: `${lo}-${hi} Gyr`,
          rows: exponentComparisons(data)
            .filter((c) => c.lo === lo && c.hi === hi)
            .map((c) => ({
              halo: c.halo,
              values: {
                measured: { value: c.measured, err: c.measuredErr },
                tracer: { value: c.tracer, err: c.tracerErr },
                sheet: { value: c.sheet, err: c.sheetErr }
              }
            }))
        }))
      }
    />
  )
}
