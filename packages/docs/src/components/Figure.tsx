import useBaseUrl from '@docusaurus/useBaseUrl'
import React, { type ReactNode } from 'react'

interface FigureProps {
  /** Path under the static directories, such as `/figures/au24-level4-Original-snap127-mono-age.png`. */
  readonly src: string
  readonly alt: string
  /** Wide figures keep the full column width. */
  readonly wide?: boolean
}

/** A static figure drawn by the Python package, centered and sized for the column. */
export default function Figure({ src, alt, wide = false }: FigureProps): ReactNode {
  return (
    <figure className={wide ? 'scientific wide' : 'scientific'}>
      <img src={useBaseUrl(src)} alt={alt} />
    </figure>
  )
}
