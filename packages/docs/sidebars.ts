import type { SidebarsConfig } from '@docusaurus/plugin-content-docs'

// Add categories as the sections of the manuscript are written.
const sidebars: SidebarsConfig = {
  docsSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Concepts',
      collapsed: false,
      items: [
        'concepts/tracer-and-sheet',
        'concepts/where-the-test-bites',
        'concepts/the-window',
        'concepts/the-potential',
        'concepts/fitting-a-population'
      ]
    },
    {
      type: 'category',
      label: 'Results',
      collapsed: false,
      items: [
        'status/overview',
        'status/tracer-vs-sheet',
        'status/window-bias',
        'status/across-galaxies',
        'status/milky-way'
      ]
    },
    {
      type: 'category',
      label: 'The disks',
      collapsed: false,
      items: ['explorations/rendered-disks', 'explorations/disk-in-3d']
    },
    'acknowledgments'
  ]
}

export default sidebars
