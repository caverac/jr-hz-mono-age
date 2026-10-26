import Link from '@docusaurus/Link'
import MathLabel from '@site/src/components/MathLabel'
import type { Props } from '@theme/PaginatorNavLink'
import clsx from 'clsx'
import React, { type ReactNode } from 'react'

/**
 * Swizzled from `@docusaurus/theme-classic` so that the previous and next
 * links can carry the inline math of a page's sidebar label. The upstream
 * component inserts the title as it stands; here a string title goes through
 * `MathLabel`.
 */
export default function PaginatorNavLink(props: Props): ReactNode {
  const { permalink, title, subLabel, isNext } = props

  return (
    <Link
      className={clsx(
        'pagination-nav__link',
        isNext ? 'pagination-nav__link--next' : 'pagination-nav__link--prev'
      )}
      to={permalink}
    >
      {subLabel && <div className="pagination-nav__sublabel">{subLabel}</div>}
      <div className="pagination-nav__label">
        {typeof title === 'string' ? <MathLabel label={title} /> : title}
      </div>
    </Link>
  )
}
