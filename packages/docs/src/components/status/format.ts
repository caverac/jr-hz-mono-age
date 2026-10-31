/**
 * The numbers that the prose of the status pages quotes, set as inline math, so that minus
 * signs and errors are typeset as in the tables and the equations around them. The metric
 * components render these strings with KaTeX.
 */

/** `value` with `digits` decimals. */
export function fixed(value: number, digits = 2): string {
  return `$${value.toFixed(digits)}$`
}

/** `value` with `digits` decimals and its sign, plus or minus; no sign when it rounds to 0. */
export function signed(value: number, digits = 2): string {
  const shown = Math.abs(value).toFixed(digits)
  const sign = Number(shown) === 0 ? '' : value >= 0 ? '+' : '-'
  return `$${sign}${shown}$`
}

/** `value` and its error `err`, with `digits` decimals; the value carries its sign if `withSign`. */
export function plusMinus(value: number, err: number, digits = 3, withSign = true): string {
  const shown = withSign
    ? `${value >= 0 ? '+' : '-'}${Math.abs(value).toFixed(digits)}`
    : value.toFixed(digits)
  // The braces keep KaTeX from breaking the line after the plus-minus sign.
  return `\${${shown} \\pm ${err.toFixed(digits)}}$`
}

/** The range from `lo` to `hi`, with `digits` decimals. */
export function between([lo, hi]: readonly [number, number], digits = 2): string {
  return `${fixed(lo, digits)} to ${fixed(hi, digits)}`
}

/** `value` as a percentage, with `digits` decimals. */
export function percent(value: number, digits = 1): string {
  return `${fixed(100 * value, digits)} percent`
}

// A bound is rounded away from the values it bounds, so that "within 1.7" holds for 1.617;
// the tolerance keeps a value that is already round, such as 2.3, from moving to 2.4.
const ROUNDING_TOLERANCE = 1e-9

/** The upper bound `value`, rounded up to `digits` decimals. */
export function upTo(value: number, digits = 2): string {
  const scale = 10 ** digits
  return fixed(Math.ceil(value * scale - ROUNDING_TOLERANCE) / scale, digits)
}

/** The lower bound `value`, rounded down to `digits` decimals. */
export function downTo(value: number, digits = 2): string {
  const scale = 10 ** digits
  return fixed(Math.floor(value * scale + ROUNDING_TOLERANCE) / scale, digits)
}

/** The upper bound `value` as a percentage, rounded up to `digits` decimals. */
export function percentUpTo(value: number, digits = 1): string {
  return `${upTo(100 * value, digits)} percent`
}

/** A count, with thousands separated by commas. */
export function count(value: number): string {
  return value.toLocaleString('en-US')
}
