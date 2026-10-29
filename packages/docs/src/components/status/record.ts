/**
 * The primitives that read the JSON records of the Python package: each checks the type of
 * one field and fails with a `RecordError` otherwise. A quantity that could not be measured
 * is `null` in the JSON and NaN here.
 */

export interface Estimate {
  readonly value: number
  readonly err: number
}

/**
 * The age exponent of a ratio h_pred / h_Z, d ln(h_pred / h_Z) / d ln age, which is 0 for a
 * prediction that reproduces the age dependence of h_Z. Its error is that of the weighted
 * fit times sqrt(max(1, chi2Dof)).
 */
export interface Trend {
  readonly value: number
  readonly err: number
  readonly chi2Dof: number
}

/** A record that does not have the expected structure. */
export class RecordError extends Error {}

export function field(value: unknown, key: string): unknown {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    throw new RecordError(`expected an object holding "${key}"`)
  }
  return (value as Record<string, unknown>)[key]
}

/** A number, or NaN where the record holds null. */
export function num(value: unknown, key: string): number {
  const item = field(value, key)
  if (item === null) {
    return Number.NaN
  }
  if (typeof item !== 'number') {
    throw new RecordError(`"${key}" is not a number`)
  }
  return item
}

export function bool(value: unknown, key: string): boolean {
  const item = field(value, key)
  if (typeof item !== 'boolean') {
    throw new RecordError(`"${key}" is not a boolean`)
  }
  return item
}

export function str(value: unknown, key: string): string {
  const item = field(value, key)
  if (typeof item !== 'string') {
    throw new RecordError(`"${key}" is not a string`)
  }
  return item
}

export function list(value: unknown, key: string): readonly unknown[] {
  const item = field(value, key)
  if (!Array.isArray(item)) {
    throw new RecordError(`"${key}" is not a list`)
  }
  return item as unknown[]
}

/** A list of numbers, with NaN where the record holds null. */
export function numbers(value: unknown, key: string): readonly number[] {
  return list(value, key).map((item) => {
    if (item === null) {
      return Number.NaN
    }
    if (typeof item !== 'number') {
      throw new RecordError(`"${key}" holds an item that is not a number`)
    }
    return item
  })
}

export function estimate(value: unknown, key: string): Estimate {
  const item = field(value, key)
  return { value: num(item, 'value'), err: num(item, 'err') }
}

export function trend(value: unknown, key: string): Trend {
  const item = field(value, key)
  return { value: num(item, 'value'), err: num(item, 'err'), chi2Dof: num(item, 'chi2_dof') }
}

export async function fetchJson(url: string, signal: AbortSignal): Promise<unknown> {
  const response = await fetch(url, { signal })
  if (!response.ok) {
    throw new RecordError(`cannot load ${url}: HTTP ${response.status}`)
  }
  return (await response.json()) as unknown
}
