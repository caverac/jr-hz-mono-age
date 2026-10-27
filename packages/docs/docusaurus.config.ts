import crypto from 'node:crypto'
import fs from 'node:fs'
import path from 'node:path'

import type * as Preset from '@docusaurus/preset-classic'
import type { Config } from '@docusaurus/types'
import { themes as prismThemes } from 'prism-react-renderer'
import rehypeKatex from 'rehype-katex'
import remarkMath from 'remark-math'

import remarkTocMath from './src/remark/tocMath'

type Stylesheet = NonNullable<Config['stylesheets']>[number]

/**
 * Stylesheet entry for KaTeX, derived from the copy in `node_modules` rather
 * than written out by hand. The stylesheet has to match the KaTeX that
 * `rehype-katex`, the table of contents and the sidebar labels render with, and
 * its integrity hash has to match the stylesheet, so pinning either by hand
 * invites a silent mismatch: a stale hash makes browsers refuse the file and
 * every formula on the site loses its layout.
 */
function katexStylesheet(): Stylesheet {
  const packageDir = path.dirname(require.resolve('katex/package.json'))
  const manifest = JSON.parse(fs.readFileSync(path.join(packageDir, 'package.json'), 'utf-8'))
  const css = fs.readFileSync(path.join(packageDir, 'dist', 'katex.min.css'))

  return {
    href: `https://cdn.jsdelivr.net/npm/katex@${manifest.version}/dist/katex.min.css`,
    type: 'text/css',
    integrity: `sha384-${crypto.createHash('sha384').update(css).digest('base64')}`,
    crossorigin: 'anonymous'
  }
}

const baseUrl = process.env.DOCS_BASE_URL ?? '/jr-hz-mono-age/'

const pyprojectPath = path.resolve(__dirname, '../experiments/pyproject.toml')
const pyprojectContent = fs.readFileSync(pyprojectPath, 'utf-8')
const versionMatch = pyprojectContent.match(/^version\s*=\s*"([^"]+)"/m)
const libraryVersion = versionMatch?.[1] ?? '0.0.0'

const rootPkgPath = path.resolve(__dirname, '../../package.json')
const rootPkg = JSON.parse(fs.readFileSync(rootPkgPath, 'utf-8'))
const projectVersion = rootPkg.version ?? '0.0.0'

const config: Config = {
  title: 'Mono-Age Populations',
  tagline: 'Mono-age disk populations and the age dependence of the J_R-h_Z relation',

  url: 'https://caverac.github.io',
  baseUrl,

  organizationName: 'caverac',
  projectName: 'jr-hz-mono-age',

  onBrokenLinks: 'warn',

  // Serve the local `static/` (site chrome: logo, favicon) plus the
  // repository-level `assets/` directory (generated figures and data). Files in
  // `assets/figures/*.png` are then available at `/figures/*.png`, so the docs
  // and the paper share one figure source, and the star samples of
  // `assets/data/auriga/` at `/data/auriga/`.
  staticDirectories: ['static', '../../assets'],

  favicon: 'img/logo.svg',

  customFields: {
    libraryVersion,
    projectVersion
  },

  i18n: {
    defaultLocale: 'en',
    locales: ['en']
  },

  stylesheets: [katexStylesheet()],

  markdown: {
    hooks: {
      onBrokenMarkdownLinks: 'warn'
    }
  },

  themes: [
    [
      '@easyops-cn/docusaurus-search-local',
      {
        hashed: true,
        language: ['en'],
        highlightSearchTermsOnTargetPage: true,
        explicitSearchResultPath: true,
        docsRouteBasePath: '/',
        indexBlog: false
      }
    ]
  ],

  presets: [
    [
      'classic',
      {
        docs: {
          routeBasePath: '/',
          sidebarPath: './sidebars.ts',
          // Must run before Docusaurus builds the table of contents, which the
          // regular `remarkPlugins` hook is too late for.
          beforeDefaultRemarkPlugins: [remarkTocMath],
          remarkPlugins: [remarkMath],
          rehypePlugins: [rehypeKatex]
        },
        blog: false,
        theme: {
          customCss: ['./src/css/custom.css', './src/css/viz.css']
        }
      } satisfies Preset.Options
    ]
  ],

  themeConfig: {
    navbar: {
      title: 'Mono-Age Populations',
      logo: {
        alt: 'Spiral galaxy logo',
        src: 'img/logo.svg'
      },
      items: [
        {
          // Project version tag, fed from the root package.json version the
          // config reads above. Rendered with the .version-badge class.
          type: 'html',
          position: 'left',
          value: `<span class="version-badge">v${projectVersion}</span>`
        },
        {
          type: 'docSidebar',
          sidebarId: 'docsSidebar',
          position: 'left',
          label: 'Documentation'
        },
        {
          href: 'https://github.com/caverac/jr-hz-mono-age',
          label: 'GitHub',
          position: 'right'
        }
      ]
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Docs',
          items: [
            { label: 'Introduction', to: '/' },
            { label: 'Acknowledgments', to: '/acknowledgments' }
          ]
        }
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Carlos Vera-Ciro. Built with Docusaurus.`
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['python', 'bash']
    }
  } satisfies Preset.ThemeConfig
}

export default config
