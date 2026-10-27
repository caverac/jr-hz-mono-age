import type { SidebarsConfig } from '@docusaurus/plugin-content-docs'

// Add categories as the sections of the manuscript are written.
const sidebars: SidebarsConfig = {
  docsSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Status',
      collapsed: false,
      items: ['status/overview', 'status/tracer-vs-sheet', 'status/window-bias']
    },
    {
      type: 'category',
      label: 'Explorations',
      collapsed: false,
      items: ['explorations/rendered-disks', 'explorations/disk-in-3d']
    },
    'acknowledgments'
  ]
}

export default sidebars
