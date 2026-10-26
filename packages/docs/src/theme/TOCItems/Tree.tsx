import Link from '@docusaurus/Link'
import { TOC_MATH_CLASS } from '@site/src/remark/tocMath'
import type { Props } from '@theme/TOCItems/Tree'
import katex from 'katex'
import React, { type ReactNode } from 'react'

/**
 * Swizzled from `@docusaurus/theme-classic` so the table of contents can show
 * rendered formulas. The upstream component differs only in that it injects
 * `heading.value` as it stands; here the marker spans left by
 * `src/remark/tocMath.ts` are handed to KaTeX first.
 */

const TOC_MATH_PATTERN = new RegExp(`<span class="${TOC_MATH_CLASS}">([\\s\\S]*?)</span>`, 'g')

const HTML_ENTITIES: Readonly<Record<string, string>> = {
  '&amp;': '&',
  '&lt;': '<',
  '&gt;': '>',
  '&quot;': '"',
  '&#39;': "'"
}

/**
 * Reverses the escaping Docusaurus applies when it serializes a heading, so
 * that KaTeX receives the LaTeX exactly as it was written. This matters for
 * `&`, which aligns columns, and for the comparisons `<` and `>`.
 */
function decodeEntities(value: string): string {
  return value.replace(/&(?:amp|lt|gt|quot|#39);/g, (entity) => HTML_ENTITIES[entity] ?? entity)
}

/**
 * Replaces every marked span with its KaTeX rendering. Errors are rendered
 * rather than thrown, so a malformed formula degrades to visibly red source in
 * the table of contents instead of taking the page down.
 */
function renderMath(value: string): string {
  return value.replace(TOC_MATH_PATTERN, (_match, tex: string) =>
    katex.renderToString(decodeEntities(tex), { throwOnError: false })
  )
}

function TOCItemTree({ toc, className, linkClassName, isChild }: Props): ReactNode {
  if (!toc.length) {
    return null
  }

  return (
    <ul className={isChild ? undefined : className}>
      {toc.map((heading) => (
        <li key={heading.id}>
          <Link
            to={`#${heading.id}`}
            className={linkClassName ?? undefined}
            // The markup is produced by our own pipeline, so it is safe.
            dangerouslySetInnerHTML={{ __html: renderMath(heading.value) }}
          />
          <TOCItemTree
            isChild
            toc={heading.children}
            className={className}
            linkClassName={linkClassName}
          />
        </li>
      ))}
    </ul>
  )
}

// Memo only the tree root is enough
export default React.memo(TOCItemTree)
