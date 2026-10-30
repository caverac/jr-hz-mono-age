// Run scripts/paper-numbers.ts through jiti, with the @site alias of the docs package.
const path = require('node:path')

const createJiti = require('jiti')

createJiti(__filename, { alias: { '@site': path.resolve(__dirname, '..') }, interopDefault: true })(
  './paper-numbers.ts'
)
