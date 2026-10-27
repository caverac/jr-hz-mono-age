/**
 * The random sample of star particles that `experiments auriga render-data` writes for
 * the 3D view: a little-endian binary of int16 x, y, z per star (in units of
 * `positionUnitKpc`) followed by uint16 ages (in units of `ageUnitGyr`), and a JSON
 * description that also carries the age palette of the rendered images.
 */

export interface SampleMeta {
  readonly label: string
  readonly count: number
  readonly positionUnitKpc: number
  readonly ageUnitGyr: number
  readonly ageGroupsGyr: readonly (readonly [number, number])[]
  readonly ageStopsGyr: readonly number[]
  readonly ageColors: readonly string[]
}

export interface StarSample {
  readonly meta: SampleMeta
  /** x, y, z of every star in kpc, in the disk frame (z along the spin). */
  readonly positions: Float32Array
  /** Age of every star in Gyr. */
  readonly ages: Float32Array
}

function isArrayOf<T>(value: unknown, isItem: (item: unknown) => item is T): value is T[] {
  return Array.isArray(value) && (value as unknown[]).every(isItem)
}

function isNumber(value: unknown): value is number {
  return typeof value === 'number'
}

function isString(value: unknown): value is string {
  return typeof value === 'string'
}

function isPair(value: unknown): value is [number, number] {
  return isArrayOf(value, isNumber) && value.length === 2
}

/** Check the JSON description, so that a file written by other code fails here and not in the shader. */
export function parseMeta(value: unknown): SampleMeta {
  if (typeof value !== 'object' || value === null) {
    throw new Error('the star sample description is not a JSON object')
  }
  const { label, count, positionUnitKpc, ageUnitGyr, ageGroupsGyr, ageStopsGyr, ageColors } =
    value as Record<string, unknown>
  if (
    !isString(label) ||
    !isNumber(count) ||
    !isNumber(positionUnitKpc) ||
    !isNumber(ageUnitGyr) ||
    !isArrayOf(ageGroupsGyr, isPair) ||
    !isArrayOf(ageStopsGyr, isNumber) ||
    !isArrayOf(ageColors, isString) ||
    ageColors.length !== ageStopsGyr.length
  ) {
    throw new Error('the star sample description lacks a field or has one of the wrong type')
  }
  return { label, count, positionUnitKpc, ageUnitGyr, ageGroupsGyr, ageStopsGyr, ageColors }
}

/** Decode the binary of `meta.count` stars into positions (kpc) and ages (Gyr). */
export function decodeSample(meta: SampleMeta, buffer: ArrayBuffer): StarSample {
  const expected = meta.count * 8
  if (buffer.byteLength !== expected) {
    throw new Error(`the star sample holds ${buffer.byteLength} bytes, not ${expected}`)
  }
  const view = new DataView(buffer)
  const positions = new Float32Array(meta.count * 3)
  const ages = new Float32Array(meta.count)
  for (let i = 0; i < meta.count * 3; i += 1) {
    positions[i] = view.getInt16(2 * i, true) * meta.positionUnitKpc
  }
  const offset = meta.count * 6
  for (let i = 0; i < meta.count; i += 1) {
    ages[i] = view.getUint16(offset + 2 * i, true) * meta.ageUnitGyr
  }
  return { meta, positions, ages }
}

/** Fetch the description and the binary at `url` (without extension) and decode them. */
export async function loadSample(url: string, signal: AbortSignal): Promise<StarSample> {
  const [metaResponse, binResponse] = await Promise.all([
    fetch(`${url}.json`, { signal }),
    fetch(`${url}.bin`, { signal })
  ])
  if (!metaResponse.ok || !binResponse.ok) {
    throw new Error(`cannot load ${url}: HTTP ${metaResponse.status} and ${binResponse.status}`)
  }
  const meta = parseMeta(await metaResponse.json())
  return decodeSample(meta, await binResponse.arrayBuffer())
}
