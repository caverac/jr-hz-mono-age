import useBaseUrl from '@docusaurus/useBaseUrl'
import { type Halo, samplePath } from '@site/src/components/render/halos'
import { parseMeta, type SampleMeta } from '@site/src/components/render/stars'
import { useEffect, useState } from 'react'

/** A value that loads in the browser: null until it arrives, with the error if it failed. */
export interface Loaded<T> {
  readonly value: T | null
  readonly error: string | null
}

/** The description of the star sample of `halo`, which carries the age palette. */
export default function useSampleMeta(halo: Halo): Loaded<SampleMeta> {
  const url = useBaseUrl(samplePath(halo))
  const [loaded, setLoaded] = useState<Loaded<SampleMeta>>({ value: null, error: null })
  useEffect(() => {
    const controller = new AbortController()
    fetch(`${url}.json`, { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) {
          throw new Error(`cannot load ${url}.json: HTTP ${response.status}`)
        }
        setLoaded({ value: parseMeta(await response.json()), error: null })
      })
      .catch((error: unknown) => {
        if (!controller.signal.aborted) {
          setLoaded({ value: null, error: error instanceof Error ? error.message : String(error) })
        }
      })
    return () => controller.abort()
  }, [url])
  return loaded
}
