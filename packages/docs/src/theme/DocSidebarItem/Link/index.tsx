import isInternalUrl from '@docusaurus/isInternalUrl'
import Link from '@docusaurus/Link'
import { isActiveSidebarItem } from '@docusaurus/plugin-content-docs/client'
import { ThemeClassNames } from '@docusaurus/theme-common'
import MathLabel, { plainLabel } from '@site/src/components/MathLabel'
import type { Props } from '@theme/DocSidebarItem/Link'
import IconExternalLink from '@theme/Icon/ExternalLink'
import clsx from 'clsx'
import React, { type ReactNode } from 'react'

import styles from './styles.module.css'

/**
 * Swizzled from `@docusaurus/theme-classic` so that a sidebar label can carry
 * inline math. The upstream component inserts the label as plain text; here it
 * goes through `MathLabel`, and the `title` attribute gets the bare LaTeX.
 */

function LinkLabel({ label }: { label: string }): ReactNode {
  return (
    <span title={plainLabel(label)} className={styles.linkLabel}>
      <MathLabel label={label} />
    </span>
  )
}

export default function DocSidebarItemLink({
  item,
  onItemClick,
  activePath,
  level,
  index: _index,
  ...props
}: Props): ReactNode {
  const { href, label, className, autoAddBaseUrl } = item
  const isActive = isActiveSidebarItem(item, activePath)
  const isInternalLink = isInternalUrl(href)

  return (
    <li
      className={clsx(
        ThemeClassNames.docs.docSidebarItemLink,
        ThemeClassNames.docs.docSidebarItemLinkLevel(level),
        'menu__list-item',
        className
      )}
      key={label}
    >
      <Link
        className={clsx('menu__link', !isInternalLink && styles.menuExternalLink, {
          'menu__link--active': isActive
        })}
        autoAddBaseUrl={autoAddBaseUrl}
        aria-current={isActive ? 'page' : undefined}
        to={href}
        {...(isInternalLink && {
          onClick: onItemClick ? () => onItemClick(item) : undefined
        })}
        {...props}
      >
        <LinkLabel label={label} />
        {!isInternalLink && <IconExternalLink />}
      </Link>
    </li>
  )
}
