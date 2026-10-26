import Link from '@docusaurus/Link'
import { useSidebarBreadcrumbs } from '@docusaurus/plugin-content-docs/client'
import { ThemeClassNames } from '@docusaurus/theme-common'
import { useHomePageRoute } from '@docusaurus/theme-common/internal'
import { translate } from '@docusaurus/Translate'
import MathLabel from '@site/src/components/MathLabel'
import HomeBreadcrumbItem from '@theme/DocBreadcrumbs/Items/Home'
import DocBreadcrumbsStructuredData from '@theme/DocBreadcrumbs/StructuredData'
import clsx from 'clsx'
import React, { type ReactNode } from 'react'

import styles from './styles.module.css'

/**
 * Swizzled from `@docusaurus/theme-classic` so that a breadcrumb can carry the
 * inline math of its sidebar label. The upstream component inserts the label as
 * plain text; here it goes through `MathLabel`.
 */

function BreadcrumbsItemLink({
  children,
  href,
  isLast
}: {
  children: ReactNode
  href: string | undefined
  isLast: boolean
}): ReactNode {
  const className = 'breadcrumbs__link'

  if (isLast) {
    return <span className={className}>{children}</span>
  }

  return href ? (
    <Link className={className} href={href}>
      <span>{children}</span>
    </Link>
  ) : (
    <span className={className}>{children}</span>
  )
}

function BreadcrumbsItem({
  children,
  active
}: {
  children: ReactNode
  active?: boolean
}): ReactNode {
  return (
    <li
      className={clsx('breadcrumbs__item', {
        'breadcrumbs__item--active': active
      })}
    >
      {children}
    </li>
  )
}

export default function DocBreadcrumbs(): ReactNode {
  const breadcrumbs = useSidebarBreadcrumbs()
  const homePageRoute = useHomePageRoute()

  if (!breadcrumbs) {
    return null
  }

  return (
    <>
      <DocBreadcrumbsStructuredData breadcrumbs={breadcrumbs} />
      <nav
        className={clsx(ThemeClassNames.docs.docBreadcrumbs, styles.breadcrumbsContainer)}
        aria-label={translate({
          id: 'theme.docs.breadcrumbs.navAriaLabel',
          message: 'Breadcrumbs',
          description: 'The ARIA label for the breadcrumbs'
        })}
      >
        <ul className="breadcrumbs">
          {homePageRoute && <HomeBreadcrumbItem />}
          {breadcrumbs.map((item, idx) => {
            const isLast = idx === breadcrumbs.length - 1
            const href = item.type === 'category' && item.linkUnlisted ? undefined : item.href

            return (
              <BreadcrumbsItem key={idx} active={isLast}>
                <BreadcrumbsItemLink href={href} isLast={isLast}>
                  <MathLabel label={item.label} />
                </BreadcrumbsItemLink>
              </BreadcrumbsItem>
            )
          })}
        </ul>
      </nav>
    </>
  )
}
