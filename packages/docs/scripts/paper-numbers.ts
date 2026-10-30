/**
 * Write the numbers of the paper as LaTeX macros, from the same functions that the status
 * pages quote (`statusMetrics.ts`, `recordMetrics.ts`), so that the paper cannot drift from
 * the records or from the pages. Each metric `name` becomes the macro `\name`, with the
 * digits of the name spelled out (`tracerWithin10` is `\tracerWithinTen`), because a LaTeX
 * macro name has letters only. The macros go into the block between `% BEGIN NUMBERS` and
 * `% END NUMBERS` of ms.tex, which keeps the manuscript a single file for the journal, and
 * only those that the text calls are written. Run it from the docs package:
 *
 *     yarn paper:numbers
 */

import { readFileSync, writeFileSync } from 'node:fs'
import path from 'node:path'

import { loadStatusData } from '@site/src/components/status/data'
import { parseJrRelation } from '@site/src/components/status/jrRelation'
import { parseMilkyWay } from '@site/src/components/status/milkyWay'
import { parseMwChecks } from '@site/src/components/status/mwChecks'
import { CHECK_METRICS, JR_METRICS, MW_METRICS } from '@site/src/components/status/recordMetrics'
import { STATUS_METRICS } from '@site/src/components/status/statusMetrics'

const REPO = path.resolve(__dirname, '..', '..', '..')
const ASSETS = path.join(REPO, 'assets')
const MANUSCRIPT = path.join(REPO, 'packages', 'pre-print', 'ms.tex')
const BEGIN = '% BEGIN NUMBERS\n'
const END = '% END NUMBERS\n'

const ONES = [
  'Zero',
  'One',
  'Two',
  'Three',
  'Four',
  'Five',
  'Six',
  'Seven',
  'Eight',
  'Nine',
  'Ten',
  'Eleven',
  'Twelve',
  'Thirteen',
  'Fourteen',
  'Fifteen',
  'Sixteen',
  'Seventeen',
  'Eighteen',
  'Nineteen'
]
const TENS = ['', '', 'Twenty', 'Thirty', 'Forty', 'Fifty', 'Sixty', 'Seventy', 'Eighty', 'Ninety']

/** `n` (0 to 99) in words, capitalized. */
function numberWord(n: number): string {
  if (n < 20) {
    return ONES[n]
  }
  if (n < 100) {
    return TENS[Math.floor(n / 10)] + (n % 10 === 0 ? '' : ONES[n % 10])
  }
  throw new Error(`no word for ${n}`)
}

/** The LaTeX macro of the metric `name`: its digits spelled out, and letters only. */
export function macroName(name: string): string {
  // A run of three or more digits, such as the 810 of 8-10 Gyr, is spelled digit by digit.
  const macro = name.replace(/\d+/g, (digits) =>
    digits.length > 2
      ? [...digits].map((digit) => numberWord(Number(digit))).join('')
      : numberWord(Number(digits))
  )
  if (!/^[A-Za-z]+$/.test(macro)) {
    throw new Error(`the metric ${name} gives the macro ${macro}, which is not letters only`)
  }
  return macro
}

/** The value of a metric as LaTeX: the inline math of the pages is LaTeX, so it only has to be ASCII. */
function latexValue(name: string, value: string): string {
  if (!/^[\x20-\x7e]*$/.test(value)) {
    throw new Error(`the value of ${name}, ${value}, is not ASCII`)
  }
  return value
}

/** Every metric of `metrics`, evaluated on `data`, by macro name. */
function evaluate<T>(
  metrics: Readonly<Record<string, (data: T) => string>>,
  data: T
): readonly (readonly [string, string])[] {
  return Object.entries(metrics).map(([name, metric]) => [
    macroName(name),
    latexValue(name, metric(data))
  ])
}

/** The text of `tex` before and after its block of numbers, which must appear once. */
function aroundNumbers(tex: string): readonly [string, string] {
  const begin = tex.indexOf(BEGIN)
  const end = tex.indexOf(END)
  if (begin < 0 || end < begin || tex.indexOf(BEGIN, begin + 1) >= 0) {
    throw new Error(`ms.tex needs one block of numbers, from "${BEGIN.trim()}" to "${END.trim()}"`)
  }
  return [tex.slice(0, begin), tex.slice(end + END.length)]
}

/** The names of the macros that `text` calls. */
function calledMacros(text: string): ReadonlySet<string> {
  return new Set([...text.matchAll(/\\([A-Za-z]+)/g)].map((match) => match[1]))
}

/** The JSON record at `relative` under the assets folder. */
function record(relative: string): unknown {
  return JSON.parse(readFileSync(path.join(ASSETS, relative), 'utf8')) as unknown
}

async function main(): Promise<void> {
  // The pages fetch the records from the site; here the site's root is the assets folder.
  globalThis.fetch = (async (input: RequestInfo | URL) =>
    new Response(readFileSync(String(input)))) as typeof fetch
  const status = await loadStatusData(`${ASSETS}/`, new AbortController().signal)
  const macros = [
    ...evaluate(STATUS_METRICS, status),
    ...evaluate(MW_METRICS, parseMilkyWay(record('data/milky-way/milky-way.json'))),
    ...evaluate(JR_METRICS, parseJrRelation(record('data/auriga/jr-relation.json'))),
    ...evaluate(CHECK_METRICS, parseMwChecks(record('data/milky-way/checks.json')))
  ]
  const seen = new Set<string>()
  for (const [macro] of macros) {
    if (seen.has(macro)) {
      throw new Error(`two metrics give the macro ${macro}`)
    }
    seen.add(macro)
  }
  const [head, tail] = aroundNumbers(readFileSync(MANUSCRIPT, 'utf8'))
  const called = calledMacros(head + tail)
  const lines = macros
    .filter(([macro]) => called.has(macro))
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([macro, value]) => `\\newcommand{\\${macro}}{${value}}`)
  writeFileSync(
    MANUSCRIPT,
    [
      head + BEGIN.trimEnd(),
      '% Written by `yarn paper:numbers` in packages/docs from the records under assets/data/,',
      '% with only the numbers that the text calls. Do not edit by hand.',
      ...lines,
      END + tail
    ].join('\n')
  )
  process.stdout.write(`wrote ${lines.length} of ${macros.length} numbers to ${MANUSCRIPT}\n`)
}

main().catch((error: unknown) => {
  process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`)
  process.exitCode = 1
})
