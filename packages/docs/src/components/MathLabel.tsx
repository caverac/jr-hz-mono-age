import katex from 'katex'
import React, { type ReactNode } from 'react'

/**
 * Splits a label on its inline math, keeping the `$...$` segments in the result.
 */
const INLINE_MATH_SPLIT = /(\$[^$]+\$)/

function isInlineMath(segment: string): boolean {
  return segment.length > 2 && segment.startsWith('$') && segment.endsWith('$')
}

/**
 * The label with its inline math reduced to the bare LaTeX, for attributes such
 * as `title` that cannot carry markup.
 */
export function plainLabel(label: string): string {
  return label
    .split(INLINE_MATH_SPLIT)
    .map((segment) => (isInlineMath(segment) ? segment.slice(1, -1) : segment))
    .join('')
}

/**
 * Renders a sidebar, breadcrumb or paginator label whose frontmatter carries
 * inline math between dollar signs. Docusaurus inserts such labels as plain
 * text, so the LaTeX would show as written; here every `$...$` segment is
 * handed to KaTeX and the rest is left as text. Errors are rendered rather than
 * thrown, so a malformed formula degrades to visibly red source.
 */
export default function MathLabel({ label }: { label: string }): ReactNode {
  return (
    <>
      {label.split(INLINE_MATH_SPLIT).map((segment, index) =>
        isInlineMath(segment) ? (
          <span
            key={index}
            // The markup comes from KaTeX, fed with our own frontmatter.
            dangerouslySetInnerHTML={{
              __html: katex.renderToString(segment.slice(1, -1), { throwOnError: false })
            }}
          />
        ) : (
          <React.Fragment key={index}>{segment}</React.Fragment>
        )
      )}
    </>
  )
}
