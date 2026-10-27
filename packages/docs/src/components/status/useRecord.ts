import useBaseUrl from '@docusaurus/useBaseUrl'
import type { Loaded } from '@site/src/components/render/useSampleMeta'
import { fetchJson } from '@site/src/components/status/data'
import { useEffect, useState } from 'react'

/** One load of each record per page, shared by every chart, table and number that reads it. */
const loads = new Map<string, Promise<unknown>>()

function load<T>(url: string, parse: (record: unknown) => T): Promise<T> {
  const cached = loads.get(url)
  if (cached !== undefined) {
    return cached as Promise<T>
  }
  const promise = fetchJson(url, new AbortController().signal).then(parse)
  loads.set(url, promise)
  promise.catch(() => loads.delete(url))
  return promise
}

/**
 * The record at `path` under the site's root, checked and read by `parse`; null until it
 * has loaded.
 */
export default function useRecord<T>(path: string, parse: (record: unknown) => T): Loaded<T> {
  const url = useBaseUrl(path)
  const [loaded, setLoaded] = useState<Loaded<T>>({ value: null, error: null })
  useEffect(() => {
    let active = true
    load(url, parse)
      .then((value) => {
        if (active) {
          setLoaded({ value, error: null })
        }
      })
      .catch((error: unknown) => {
        if (active) {
          setLoaded({ value: null, error: error instanceof Error ? error.message : String(error) })
        }
      })
    return () => {
      active = false
    }
  }, [url, parse])
  return loaded
}
