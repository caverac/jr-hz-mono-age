import useBaseUrl from '@docusaurus/useBaseUrl'
import AgeKey from '@site/src/components/render/AgeKey'
import { DEFAULT_HALO, HALOS, portraitPath } from '@site/src/components/render/halos'
import useSampleMeta from '@site/src/components/render/useSampleMeta'
import { useElementWidth } from '@site/src/components/viz/hooks'
import React, { type KeyboardEvent, type ReactNode, useState } from 'react'

/**
 * The portraits that `experiments auriga render` draws, one at a time on a black stage
 * with the others as thumbnails. Click a thumbnail, or use the arrow keys, to change the
 * halo on the stage.
 */
export default function GalaxyGallery(): ReactNode {
  const [containerRef, width] = useElementWidth<HTMLDivElement>()
  const [selected, setSelected] = useState(DEFAULT_HALO)
  const root = useBaseUrl('/')
  const meta = useSampleMeta(HALOS.find((h) => h.label === DEFAULT_HALO) ?? HALOS[0]).value
  const src = (path: string): string => `${root}${path.replace(/^\//, '')}`

  const step = (offset: number): void => {
    const index = HALOS.findIndex((h) => h.label === selected)
    const next = HALOS[(index + offset + HALOS.length) % HALOS.length]
    setSelected(next.label)
  }
  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>): void => {
    if (event.key === 'ArrowRight' || event.key === 'ArrowDown') {
      event.preventDefault()
      step(1)
    } else if (event.key === 'ArrowLeft' || event.key === 'ArrowUp') {
      event.preventDefault()
      step(-1)
    }
  }

  return (
    <div className="render" ref={containerRef}>
      <div
        className="gallery"
        tabIndex={0}
        onKeyDown={onKeyDown}
        role="group"
        aria-label="Portraits of the seven Auriga disks; the arrow keys change the halo"
      >
        <div className="gallery__stage">
          {HALOS.map((h) => (
            <img
              key={h.label}
              src={src(portraitPath(h))}
              alt={`${h.label} face-on and edge-on, colored by stellar age`}
              data-shown={h.label === selected}
              aria-hidden={h.label !== selected}
            />
          ))}
        </div>
        <div className="gallery__thumbs">
          {HALOS.map((h) => (
            <button
              key={h.label}
              type="button"
              className="gallery__thumb"
              aria-pressed={h.label === selected}
              onClick={() => setSelected(h.label)}
            >
              <img src={src(portraitPath(h))} alt="" loading="lazy" />
              <span>{h.label}</span>
            </button>
          ))}
        </div>
      </div>
      {meta === null ? null : (
        <div className="render__bar">
          <AgeKey
            width={Math.max(width - 30, 200)}
            stops={meta.ageStopsGyr}
            colors={meta.ageColors}
          />
        </div>
      )}
    </div>
  )
}
