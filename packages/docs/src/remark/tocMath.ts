/**
 * Class applied to the span this plugin wraps around inline math in a heading.
 * The swizzled table of contents looks for it, see `src/theme/TOCItems/Tree.tsx`.
 */
export const TOC_MATH_CLASS = 'toc-math'

interface MdastNode {
  type: string
  children?: MdastNode[]
}

interface MdxJsxAttribute {
  type: 'mdxJsxAttribute'
  name: string
  value: string
}

interface MdxJsxTextElement extends MdastNode {
  type: 'mdxJsxTextElement'
  name: string
  attributes: MdxJsxAttribute[]
  children: MdastNode[]
}

/**
 * Wraps a math node in a marked span, leaving anything else untouched but
 * descending into it, since a heading may nest math inside emphasis or a link.
 */
function wrapInlineMath(node: MdastNode): MdastNode {
  if (node.type === 'inlineMath') {
    const wrapper: MdxJsxTextElement = {
      type: 'mdxJsxTextElement',
      name: 'span',
      attributes: [{ type: 'mdxJsxAttribute', name: 'className', value: TOC_MATH_CLASS }],
      children: [node]
    }

    return wrapper
  }

  if (node.children !== undefined) {
    node.children = node.children.map(wrapInlineMath)
  }

  return node
}

function markHeadings(node: MdastNode): void {
  if (node.children === undefined) {
    return
  }

  if (node.type === 'heading') {
    node.children = node.children.map(wrapInlineMath)
    return
  }

  for (const child of node.children) {
    markHeadings(child)
  }
}

/**
 * Makes the LaTeX in a heading survive into the table of contents.
 *
 * Docusaurus derives the table of contents from the Markdown syntax tree, and
 * its serializer has no case for math, so an `inlineMath` node falls through to
 * a plain stringification of its source and the table of contents shows the
 * LaTeX rather than the formula. Wrapping the math in a JSX span gives that
 * serializer a tag it does handle, so the entry reaches the browser carrying a
 * marker the theme can hand to KaTeX.
 *
 * The wrapper is inert in the heading itself: the original math node is kept as
 * its child, so `remark-math` and `rehype-katex` render the page as before.
 *
 * Register this through `beforeDefaultRemarkPlugins`. Anything passed as
 * `remarkPlugins` runs after Docusaurus has already built the table of
 * contents, which is too late to affect it. The same approach is used in the
 * `qft-bogolubov` docs.
 */
export default function remarkTocMath(): (root: MdastNode) => void {
  return (root) => {
    markHeadings(root)
  }
}
