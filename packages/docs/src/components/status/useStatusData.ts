import useBaseUrl from '@docusaurus/useBaseUrl'
import type { Loaded } from '@site/src/components/render/useSampleMeta'
import { loadStatusData, type StatusData } from '@site/src/components/status/data'
import { useEffect, useState } from 'react'

/** One load of the records per page, shared by every chart, table and number on it. */
const loads = new Map<string, Promise<StatusData>>()

function load(root: string): Promise<StatusData> {
  const cached = loads.get(root)
  if (cached !== undefined) {
    return cached
  }
  const promise = loadStatusData(root, new AbortController().signal)
  loads.set(root, promise)
  promise.catch(() => loads.delete(root))
  return promise
}

/** The records of the seven halos and of the injection test; null until they have loaded. */
export default function useStatusData(): Loaded<StatusData> {
  const root = useBaseUrl('/')
  const [loaded, setLoaded] = useState<Loaded<StatusData>>({ value: null, error: null })
  useEffect(() => {
    let active = true
    load(root)
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
  }, [root])
  return loaded
}
