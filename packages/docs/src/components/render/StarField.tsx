import useBaseUrl from '@docusaurus/useBaseUrl'
import AgeKey, { AGE_DOMAIN, isWholeDomain } from '@site/src/components/render/AgeKey'
import { DEFAULT_HALO, HALOS, samplePath } from '@site/src/components/render/halos'
import ScaleBar, { pixelsPerKpc } from '@site/src/components/render/ScaleBar'
import { loadSample, type SampleMeta, type StarSample } from '@site/src/components/render/stars'
import StarScene, { FIELD_OF_VIEW_DEG, type View } from '@site/src/components/render/starScene'
import type { Loaded } from '@site/src/components/render/useSampleMeta'
import { useElementWidth, useInView, usePrefersReducedMotion } from '@site/src/components/viz/hooks'
import * as d3 from 'd3'
import React, { type ReactNode, useCallback, useEffect, useRef, useState } from 'react'

type Range = readonly [number, number]

const DISTANCE_KPC: readonly [number, number, number] = [20, 62, 140]
const VIEWS: readonly (readonly [View, string])[] = [
  ['face', 'face-on'],
  ['tilt', 'tilted'],
  ['edge', 'edge-on']
]

/** The height of the view for a container `width` px wide. */
function heightFor(width: number): number {
  return Math.round(Math.min(width * 0.62, 620))
}

/** Whether two ranges of ages are the same to 0.01 Gyr. */
function sameRange(a: Range, b: Range): boolean {
  return Math.abs(a[0] - b[0]) < 0.01 && Math.abs(a[1] - b[1]) < 0.01
}

interface AgePanelProps {
  readonly width: number
  readonly meta: SampleMeta
  readonly range: Range
  readonly onRange: (range: Range) => void
}

/** The age key, which is also a brush, and one button per age group of the rendered images. */
function AgePanel({ width, meta, range, onRange }: AgePanelProps): ReactNode {
  const colorAt = d3
    .scaleLinear<string>()
    .domain([...meta.ageStopsGyr])
    .range([...meta.ageColors])
    .clamp(true)
  const groups = meta.ageGroupsGyr.map(([lo, hi]) => ({
    label: `${lo}-${hi} Gyr`,
    range: [lo, Math.min(hi, AGE_DOMAIN[1])] as const,
    color: colorAt((lo + hi) / 2)
  }))
  return (
    <div className="render__panel">
      <AgeKey
        width={Math.max(width - 30, 200)}
        stops={meta.ageStopsGyr}
        colors={meta.ageColors}
        range={range}
        onRange={onRange}
      />
      <div className="render__chips" role="group" aria-label="age groups">
        <button
          type="button"
          className="render__chip"
          aria-pressed={isWholeDomain(range)}
          onClick={() => onRange(AGE_DOMAIN)}
        >
          all ages
        </button>
        {groups.map((group) => (
          <button
            key={group.label}
            type="button"
            className="render__chip"
            aria-pressed={sameRange(range, group.range)}
            onClick={() => onRange(group.range)}
          >
            <span
              className="render__dot"
              style={{ background: group.color, boxShadow: `0 0 6px ${group.color}` }}
            />
            {group.label}
          </button>
        ))}
        <span className="render__value">
          showing {range[0].toFixed(1)} to {range[1].toFixed(1)} Gyr
        </span>
      </div>
    </div>
  )
}

/**
 * The star particles of one Auriga disk in three dimensions, colored by age. The
 * controls on the view act on the camera (halo, view, rotation, distance); the panel
 * below it chooses the ages shown.
 */
export default function StarField(): ReactNode {
  const [containerRef, width] = useElementWidth<HTMLDivElement>()
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const sceneRef = useRef<StarScene | null>(null)
  const inView = useInView(containerRef)
  const reducedMotion = usePrefersReducedMotion()
  const [haloLabel, setHaloLabel] = useState(DEFAULT_HALO)
  const halo = HALOS.find((h) => h.label === haloLabel) ?? HALOS[0]
  const sampleUrl = useBaseUrl(samplePath(halo))
  const [sample, setSample] = useState<Loaded<StarSample>>({ value: null, error: null })
  const [ageRange, setAgeRange] = useState<Range>(AGE_DOMAIN)
  const [view, setView] = useState<View>('tilt')
  const [distance, setDistance] = useState(DISTANCE_KPC[1])
  const [rotate, setRotate] = useState(true)
  const [webglError, setWebglError] = useState<string | null>(null)
  const height = heightFor(width)
  const onRange = useCallback((range: Range) => setAgeRange(range), [])

  useEffect(() => {
    const canvas = canvasRef.current
    if (canvas === null) {
      return undefined
    }
    try {
      sceneRef.current = new StarScene(canvas, DISTANCE_KPC[1])
    } catch (error: unknown) {
      setWebglError(error instanceof Error ? error.message : String(error))
      return undefined
    }
    return () => {
      sceneRef.current?.dispose()
      sceneRef.current = null
    }
  }, [])

  useEffect(() => {
    const controller = new AbortController()
    setSample({ value: null, error: null })
    loadSample(sampleUrl, controller.signal)
      .then((loaded) => setSample({ value: loaded, error: null }))
      .catch((error: unknown) => {
        if (!controller.signal.aborted) {
          setSample({ value: null, error: error instanceof Error ? error.message : String(error) })
        }
      })
    return () => controller.abort()
  }, [sampleUrl])

  useEffect(() => {
    if (sample.value !== null) {
      sceneRef.current?.setSample(sample.value)
    }
  }, [sample])

  useEffect(() => sceneRef.current?.setSize(width, height), [width, height])
  useEffect(() => sceneRef.current?.setAgeRange(ageRange), [ageRange])
  useEffect(() => sceneRef.current?.setDistance(distance), [distance])
  useEffect(() => sceneRef.current?.setView(view, !reducedMotion), [view, reducedMotion])
  useEffect(() => sceneRef.current?.setRotate(rotate && !reducedMotion), [rotate, reducedMotion])
  useEffect(() => {
    const scene = sceneRef.current
    if (scene === null || !inView) {
      return undefined
    }
    scene.start()
    return () => scene.stop()
  }, [inView])

  const status =
    webglError ?? sample.error ?? (sample.value === null ? `loading ${halo.label}` : null)
  const meta = sample.value?.meta
  return (
    <div className="render" ref={containerRef}>
      <div className="starfield">
        <canvas
          ref={canvasRef}
          style={{ height }}
          aria-label={`Star particles of ${halo.label} in three dimensions, colored by age`}
        />
        <div className="starfield__bar starfield__bar--top">
          <div className="starfield__group">
            <label className="render__field">
              halo
              <select value={haloLabel} onChange={(event) => setHaloLabel(event.target.value)}>
                {HALOS.map((h) => (
                  <option key={h.label} value={h.label}>
                    {h.label}
                  </option>
                ))}
              </select>
            </label>
          </div>
          <div className="starfield__group">
            <div className="render__segmented" role="group" aria-label="camera view">
              {VIEWS.map(([key, label]) => (
                <button
                  key={key}
                  type="button"
                  className="render__button"
                  aria-pressed={view === key}
                  onClick={() => setView(key)}
                >
                  {label}
                </button>
              ))}
            </div>
            <button
              type="button"
              className="render__button"
              aria-pressed={rotate}
              onClick={() => setRotate(!rotate)}
            >
              rotate
            </button>
          </div>
        </div>
        <div className="starfield__bar starfield__bar--bottom">
          <div className="starfield__group starfield__group--plain">
            <ScaleBar pxPerKpc={pixelsPerKpc(height, distance, FIELD_OF_VIEW_DEG)} />
          </div>
          <div className="starfield__group">
            <span className="starfield__hint">drag to turn</span>
            <label className="render__field">
              distance
              <input
                type="range"
                min={DISTANCE_KPC[0]}
                max={DISTANCE_KPC[2]}
                step={1}
                value={distance}
                onChange={(event) => setDistance(Number(event.target.value))}
              />
            </label>
          </div>
        </div>
        {status === null ? null : <div className="starfield__status">{status}</div>}
      </div>
      {meta === undefined ? null : (
        <AgePanel width={width} meta={meta} range={ageRange} onRange={onRange} />
      )}
    </div>
  )
}
