/**
 * The numbers of the Milky Way page and of the page across galaxies that their records give,
 * as inline math. They are plain functions of the records, so that the pages and the paper
 * (`scripts/paper-numbers.ts`) quote the same numbers.
 */

import {
  between,
  count,
  fixed,
  percent,
  plusMinus,
  signed
} from '@site/src/components/status/format'
import {
  type JrRelationData,
  type LeftOut,
  type Line
} from '@site/src/components/status/jrRelation'
import { andList, extent, median, trendText } from '@site/src/components/status/metrics'
import {
  boxText,
  type CellRatio,
  cellMedian,
  type ImigBin,
  type ImigCell,
  type ImigCells,
  type ImigHeight,
  imigAge,
  isMeasured,
  measuredIn,
  type FamilyMember,
  type Inversion,
  type MilkyWayData,
  type MwRange,
  type MwSurvey,
  type TargetedCells,
  tracerGrowth
} from '@site/src/components/status/milkyWay'
import type { MwChecks } from '@site/src/components/status/mwChecks'
import type { Estimate, Trend } from '@site/src/components/status/record'

/** The halo whose removal moves the slope with the self-consistent heights the most. */
function mostInfluential(d: JrRelationData): LeftOut {
  const shift = (left: LeftOut) => Math.abs(left.lines.lineScaled.slope - d.lineScaled.slope)
  return d.leaveOneOut.reduce((best, left) => (shift(left) > shift(best) ? left : best))
}

function lineText(line: Line): string {
  return `${plusMinus(line.slope, line.slopeErr, 0, false)} km/s`
}

/** The growth of a height with sigma_z, h_Z proportional to sigma_z^slope, with its error. */
function dispersionText(slope: Trend): string {
  return `$\\sigma_z^{${slope.value.toFixed(2)} \\pm ${slope.err.toFixed(2)}}$`
}

/** The surveys, in the order in which the prose names them: LAMOST, APOGEE on the astroNN and the distmass scale. */
const MW_SURVEYS = ['lamost', 'apogee', 'apogee_distmass'] as const

/** eta with its error, as a fit of the heights against sigma_z gives it. */
function etaText(eta: Trend): string {
  return plusMinus(eta.value, eta.err, 2, false)
}

/** One quantity of the inversion of every survey with its error, in the order of the surveys, joined by commas. */
function inversionText(
  inversions: readonly Inversion[],
  of: (inv: Inversion) => Estimate,
  digits: number
): string {
  return andList(inversions.map((inv) => plusMinus(of(inv).value, of(inv).err, digits, false)))
}

/** The potentials of the family within one error of the measured rho_0 and Sigma_1.1. */
function admissible(data: MilkyWayData): readonly FamilyMember[] {
  return data.family.filter((m) => m.admissible)
}

/** The member of the family without changes, which is McMillan (2017) itself. */
function mcmillan(data: MilkyWayData): FamilyMember {
  const found = data.family.find(
    (m) => m.thinFactor === 1 && m.thinHeight === 0.3 && m.haloFactor === 1
  )
  if (found === undefined) {
    throw new Error('the family has no member without changes')
  }
  return found
}

function survey(data: MilkyWayData, name: string): MwSurvey {
  const found = data.surveys.find((s) => s.name === name)
  if (found === undefined) {
    throw new Error(`no survey ${name}`)
  }
  return found
}

function range(data: MilkyWayData, name: string, lo: number): MwRange {
  const found = survey(data, name).ranges.find((r) => r.lo === lo)
  if (found === undefined) {
    throw new Error(`no range from ${lo} Gyr in ${name}`)
  }
  return found
}

function ratioRange(data: MilkyWayData, name: string, which: 'ratioTracer' | 'ratioSheet'): string {
  return between(extent(survey(data, name).bins.map((b) => b[which].value)))
}

function within10(data: MilkyWayData, name: string): string {
  const ratios = survey(data, name).bins.map((b) => b.ratioTracer.value)
  return `${ratios.filter((r) => Math.abs(r - 1) <= 0.1).length} of ${ratios.length}`
}

function countHeight(data: MilkyWayData, name: string, ageLo: number): string {
  const bin = survey(data, name).bins.find((b) => b.ageLo === ageLo)
  return bin === undefined
    ? 'none'
    : `${plusMinus(bin.measured.value, bin.measured.err, 3, false)} kpc`
}

/** The Imig et al. bins whose mean age lies in [lo, hi) Gyr. */
function imigBins(data: MilkyWayData, lo = 0, hi = Number.POSITIVE_INFINITY): readonly ImigBin[] {
  return data.imig.bins.filter((b) => imigAge(b) >= lo && imigAge(b) < hi)
}

/** The range of the height `top` over the height `bottom` in the Imig et al. bins of mean age in [lo, hi) Gyr. */
function heightRatios(
  data: MilkyWayData,
  top: ImigHeight,
  bottom: ImigHeight,
  lo?: number,
  hi?: number
): string {
  return between(
    extent(imigBins(data, lo, hi).map((b) => b.heights[top].value / b.heights[bottom].value))
  )
}

/** The range of the tracer over the height `height` in the Imig et al. bins. */
function imigRatios(data: MilkyWayData, height: ImigHeight): string {
  return between(extent(data.imig.bins.map((b) => b.ratios[height].value)))
}

/** The Imig et al. bin that holds `age` Gyr, if any. */
function imigBinAt(data: MilkyWayData, age: number): ImigBin | undefined {
  return data.imig.bins.find((b) => b.ageLo <= age && age < b.ageHi)
}

/** The height `height` of the Imig et al. bin that holds `age` Gyr. */
function imigHeight(data: MilkyWayData, height: ImigHeight, age: number): string {
  const bin = imigBinAt(data, age)
  return bin === undefined
    ? 'none'
    : `${plusMinus(bin.heights[height].value, bin.heights[height].err, 3, false)} kpc`
}

/** The median of the ratio `ratio` in the measured `cells` whose [M/H] center lies in [lo, hi), and their number. */
function medianText(
  cells: ImigCells,
  lo: number,
  hi: number,
  ratio: CellRatio = 'counted'
): string {
  const { median, n } = cellMedian(cells, lo, hi, ratio)
  return `${fixed(median)} over ${n} cells`
}

/** The median of the ratio `ratio` in the measured APOGEE cells whose age bin lies inside [lo, hi) Gyr, and their number. */
function ageMedianText(data: MilkyWayData, ratio: CellRatio, lo: number, hi: number): string {
  const values = data.imigCells.cells
    .filter((c) => isMeasured(data.imigCells, c) && c.ageLo >= lo - 1e-9 && c.ageHi <= hi + 1e-9)
    .map((c) => c.ratios[ratio].value)
    .filter((r) => Number.isFinite(r))
  return `${fixed(median(values))} over ${values.length} cells`
}

/** The median gradient of our counts over the measured APOGEE cells whose [M/H] center lies in [lo, hi), per kpc. */
function gradientText(data: MilkyWayData, lo: number, hi: number): string {
  return `${signed(median(measuredIn(data.imigCells, lo, hi).map((c) => c.ours.gradient)))} kpc$^{-1}$`
}

/** What the counts of the survey `name` give an exponential of `scale` kpc, over the scale. */
function biasText(data: MilkyWayData, name: string, scale: number): string {
  const { scales, ratios } = survey(data, name).bias
  return fixed(ratios[scales.findIndex((s) => Math.abs(s - scale) < 1e-9)] ?? Number.NaN)
}

/** The range over the measured APOGEE cells of `value`, with three decimals. */
function cellRange(data: MilkyWayData, value: (cell: ImigCell) => number): string {
  const cells = data.imigCells.cells.filter((c) => isMeasured(data.imigCells, c))
  return between(extent(cells.map(value)), 3)
}

/** The median shape of the measured APOGEE cells with [M/H] in [lo, hi), slice by slice. */
function medianShape(data: MilkyWayData, lo: number, hi: number): readonly number[] {
  const shapes = measuredIn(data.imigCells, lo, hi).map((c) => c.ours.shape)
  return data.sliceEdges
    .slice(1)
    .map((_, i) => median(shapes.map((shape) => shape[i] ?? Number.NaN)))
}

/** The targeting box `index` of the record and its cells. */
function targeted(data: MilkyWayData, index: number): TargetedCells {
  const found = data.imigCellsTargeting[index]
  if (found === undefined) {
    throw new Error(`no targeting box ${index}`)
  }
  return found
}

export const MW_METRICS = {
  r0: (d: MilkyWayData) => fixed(d.r0, 3),
  sigmaSurf: (d: MilkyWayData) => fixed(d.sigmaSurf, 1),
  nBootstrap: (d: MilkyWayData) => String(d.nBootstrap),
  lamostGaia: (d: MilkyWayData) => count(survey(d, 'lamost').nGaia),
  lamostMatched: (d: MilkyWayData) => count(survey(d, 'lamost').nMatched),
  apogeeGaia: (d: MilkyWayData) => count(survey(d, 'apogee').nGaia),
  apogeeMatched: (d: MilkyWayData) => count(survey(d, 'apogee').nMatched),
  distmassMatched: (d: MilkyWayData) => count(survey(d, 'apogee_distmass').nMatched),
  lamostTracer: (d: MilkyWayData) => ratioRange(d, 'lamost', 'ratioTracer'),
  jeansRatios: (d: MilkyWayData) =>
    between(
      extent(MW_SURVEYS.flatMap((name) => survey(d, name).bins.map((b) => b.ratioJeans.value)))
    ),
  jeansMedians: (d: MilkyWayData) =>
    between(
      extent(MW_SURVEYS.map((name) => median(survey(d, name).bins.map((b) => b.ratioJeans.value))))
    ),
  jeansBelowOne: (d: MilkyWayData) => {
    const ratios = MW_SURVEYS.flatMap((name) => survey(d, name).bins.map((b) => b.ratioJeans.value))
    return `${ratios.filter((r) => r < 1).length} of ${ratios.length}`
  },
  jeansTrends0: (d: MilkyWayData) =>
    andList(MW_SURVEYS.map((name) => trendText(range(d, name, 0).jeans))),
  familyAdmissible: (d: MilkyWayData) =>
    `${d.family.filter((m) => m.admissible).length} of ${d.family.length}`,
  familyRho0: (d: MilkyWayData) => between(extent(admissible(d).map((m) => m.rho0)), 3),
  familySigma11: (d: MilkyWayData) => between(extent(admissible(d).map((m) => m.sigma11)), 1),
  familyVCirc: (d: MilkyWayData) => between(extent(admissible(d).map((m) => m.vCirc)), 0),
  mcmillanRho0: (d: MilkyWayData) => fixed(mcmillan(d).rho0, 3),
  mcmillanSigma11: (d: MilkyWayData) => fixed(mcmillan(d).sigma11, 1),
  mcmillanVCirc: (d: MilkyWayData) => fixed(mcmillan(d).vCirc, 0),
  familyTracerMedians: (d: MilkyWayData) =>
    between(
      extent(
        d.family.flatMap((m, i) =>
          m.admissible
            ? MW_SURVEYS.map((name) => median(survey(d, name).bins.map((b) => b.ratioFamily[i])))
            : []
        )
      )
    ),
  familyGrowthWindow: (d: MilkyWayData) =>
    between(extent(admissible(d).map((m) => m.growthWindow))),
  familyGrowthWhole: (d: MilkyWayData) => between(extent(admissible(d).map((m) => m.growthWhole))),
  coevalMedians: (d: MilkyWayData) =>
    between(
      extent(MW_SURVEYS.map((name) => median(survey(d, name).bins.map((b) => b.ratioCoeval))))
    ),
  familyCoevalMedians: (d: MilkyWayData) =>
    between(
      extent(
        d.family.flatMap((m, i) =>
          m.admissible
            ? MW_SURVEYS.map((name) =>
                median(survey(d, name).bins.map((b) => b.ratioFamilyCoeval[i]))
              )
            : []
        )
      )
    ),
  bIsoMw: (d: MilkyWayData) =>
    between(extent(MW_SURVEYS.flatMap((name) => survey(d, name).bins.map((b) => b.bIsothermal)))),
  inversionRho0: (d: MilkyWayData) => inversionText(d.inversions, (inv) => inv.rho0, 3),
  inversionSigma11: (d: MilkyWayData) => inversionText(d.inversions, (inv) => inv.sigma11, 0),
  inversionChi2: (d: MilkyWayData) => between(extent(d.inversions.map((inv) => inv.chi2Dof)), 1),
  inversionModelRms: (d: MilkyWayData) =>
    percent(Math.max(...d.inversions.map((inv) => inv.modelRms)), 1),
  inversionCoevalRho0: (d: MilkyWayData) => inversionText(d.inversionsCoeval, (inv) => inv.rho0, 3),
  inversionCoevalSigma11: (d: MilkyWayData) =>
    inversionText(d.inversionsCoeval, (inv) => inv.sigma11, 0),
  inversionCoevalChi2: (d: MilkyWayData) =>
    between(extent(d.inversionsCoeval.map((inv) => inv.chi2Dof)), 1),
  mwBetaZ: (d: MilkyWayData) =>
    between(extent(MW_SURVEYS.map((name) => range(d, name, 0).betaZ.value))),
  mwBetaH: (d: MilkyWayData) =>
    between(extent(MW_SURVEYS.map((name) => range(d, name, 0).betaH.value))),
  mwEtaBetaZ: (d: MilkyWayData) =>
    between(
      extent(
        MW_SURVEYS.map((name) => {
          const r = range(d, name, 0)
          return r.eta.value * r.betaZ.value
        })
      )
    ),
  mwTwoBetaZ: (d: MilkyWayData) =>
    between(extent(MW_SURVEYS.map((name) => 2 * range(d, name, 0).betaZ.value))),
  tracerMedians: (d: MilkyWayData) => {
    const [lo, hi] = extent(
      MW_SURVEYS.map((name) => median(survey(d, name).bins.map((b) => b.ratioTracer.value)))
    )
    return between([lo, hi])
  },
  tracerBelowOne: (d: MilkyWayData) => {
    const ratios = MW_SURVEYS.flatMap((name) =>
      survey(d, name).bins.map((b) => b.ratioTracer.value)
    )
    return `${ratios.filter((r) => r < 1).length} of ${ratios.length}`
  },
  apogeeTracer: (d: MilkyWayData) => ratioRange(d, 'apogee', 'ratioTracer'),
  distmassTracer: (d: MilkyWayData) => ratioRange(d, 'apogee_distmass', 'ratioTracer'),
  lamostTracerTrend0: (d: MilkyWayData) => trendText(range(d, 'lamost', 0).tracer),
  apogeeTracerTrend0: (d: MilkyWayData) => trendText(range(d, 'apogee', 0).tracer),
  lamostTracerTrend2: (d: MilkyWayData) => trendText(range(d, 'lamost', 2).tracer),
  apogeeTracerTrend2: (d: MilkyWayData) => trendText(range(d, 'apogee', 2).tracer),
  distmassTracerTrend0: (d: MilkyWayData) => trendText(range(d, 'apogee_distmass', 0).tracer),
  distmassTracerTrend2: (d: MilkyWayData) => trendText(range(d, 'apogee_distmass', 2).tracer),
  lamostEta: (d: MilkyWayData) => etaText(range(d, 'lamost', 0).eta),
  apogeeEta: (d: MilkyWayData) => etaText(range(d, 'apogee', 0).eta),
  distmassEta: (d: MilkyWayData) => etaText(range(d, 'apogee_distmass', 0).eta),
  lamostWithin10: (d: MilkyWayData) => within10(d, 'lamost'),
  apogeeWithin10: (d: MilkyWayData) => within10(d, 'apogee'),
  distmassWithin10: (d: MilkyWayData) => within10(d, 'apogee_distmass'),
  apogeeHz24: (d: MilkyWayData) => countHeight(d, 'apogee', 2),
  distmassFailed810: (d: MilkyWayData) => {
    const bin = survey(d, 'apogee_distmass').bins.find((b) => b.ageLo === 8)
    return bin === undefined ? 'none' : percent(bin.failed, 0)
  },
  distmassHz24: (d: MilkyWayData) => countHeight(d, 'apogee_distmass', 2),
  imigAges7: (d: MilkyWayData) => {
    const bin = imigBinAt(d, 7)
    return bin === undefined ? 'none' : `${fixed(bin.ageLo)}-${fixed(bin.ageHi)} Gyr`
  },
  imigHz7: (d: MilkyWayData) => imigHeight(d, 'imigMass', 7),
  imigCountsHz7: (d: MilkyWayData) => imigHeight(d, 'imigCounts', 7),
  countsDistmassHz7: (d: MilkyWayData) => imigHeight(d, 'countsDistmass', 7),
  imigWeighting: (d: MilkyWayData) => heightRatios(d, 'imigCounts', 'imigMass'),
  imigAgeScale: (d: MilkyWayData) => heightRatios(d, 'countsAstronn', 'countsDistmass'),
  imigGapYoung: (d: MilkyWayData) => heightRatios(d, 'countsDistmass', 'imigCounts', 0, 4),
  imigGapOld: (d: MilkyWayData) => heightRatios(d, 'countsDistmass', 'imigCounts', 4),
  imigMassRatio: (d: MilkyWayData) => imigRatios(d, 'imigMass'),
  imigCountsRatio: (d: MilkyWayData) => imigRatios(d, 'imigCounts'),
  countsDistmassRatio: (d: MilkyWayData) => imigRatios(d, 'countsDistmass'),
  imigMassTrend: (d: MilkyWayData) => trendText(d.imig.trends.imigMass),
  imigCountsTrend: (d: MilkyWayData) => trendText(d.imig.trends.imigCounts),
  countsDistmassTrend: (d: MilkyWayData) => trendText(d.imig.trends.countsDistmass),
  cellsMin: (d: MilkyWayData) => String(d.imigCells.minMatches),
  cellsMeasured: (d: MilkyWayData) => `${d.imigCells.nMeasured} of ${d.imigCells.cells.length}`,
  cellsMedian: (d: MilkyWayData) => fixed(d.imigCells.medianRatio),
  cellsTrend: (d: MilkyWayData) => trendText(d.imigCells.trends.counted.metallicity),
  cellsTrendAge: (d: MilkyWayData) => trendText(d.imigCells.trends.counted.age),
  cellsExpMedian: (d: MilkyWayData) =>
    medianText(d.imigCells, Number.NEGATIVE_INFINITY, Number.POSITIVE_INFINITY, 'exponential'),
  cellsExpPoor: (d: MilkyWayData) =>
    medianText(d.imigCells, Number.NEGATIVE_INFINITY, -0.1, 'exponential'),
  cellsExpSolar: (d: MilkyWayData) => medianText(d.imigCells, -0.1, 0.1, 'exponential'),
  cellsExpRich: (d: MilkyWayData) =>
    medianText(d.imigCells, 0.1, Number.POSITIVE_INFINITY, 'exponential'),
  cellsExpTrend: (d: MilkyWayData) => trendText(d.imigCells.trends.exponential.metallicity),
  cellsExpTrendAge: (d: MilkyWayData) => trendText(d.imigCells.trends.exponential.age),
  cellsExpYoung: (d: MilkyWayData) => ageMedianText(d, 'exponential', 0, 2.51),
  cellsExpOld: (d: MilkyWayData) => ageMedianText(d, 'exponential', 5.01, Number.POSITIVE_INFINITY),
  cellsTracerPoor: (d: MilkyWayData) =>
    medianText(d.imigCells, Number.NEGATIVE_INFINITY, -0.1, 'tracer'),
  cellsTracerSolar: (d: MilkyWayData) => medianText(d.imigCells, -0.1, 0.1, 'tracer'),
  cellsTracerRich: (d: MilkyWayData) =>
    medianText(d.imigCells, 0.1, Number.POSITIVE_INFINITY, 'tracer'),
  cellsTracerTrend: (d: MilkyWayData) => trendText(d.imigCells.trends.tracer.metallicity),
  cellsTracerTrendAge: (d: MilkyWayData) => trendText(d.imigCells.trends.tracer.age),
  cellsKinMin: (d: MilkyWayData) =>
    String(
      Math.min(
        ...d.imigCells.cells
          .filter((c) => isMeasured(d.imigCells, c) && Number.isFinite(c.ratios.tracer.value))
          .map((c) => c.kinematics.nKin)
      )
    ),
  cellsWithTracer: (d: MilkyWayData) => {
    const measured = d.imigCells.cells.filter((c) => isMeasured(d.imigCells, c))
    return `${measured.filter((c) => Number.isFinite(c.ratios.tracer.value)).length} of ${measured.length}`
  },
  gradientPoor: (d: MilkyWayData) => gradientText(d, Number.NEGATIVE_INFINITY, -0.1),
  gradientRich: (d: MilkyWayData) => gradientText(d, 0.1, Number.POSITIVE_INFINITY),
  lamostExpTrendAge: (d: MilkyWayData) => trendText(d.imigCellsLamost.trends.exponential.age),
  cellsMetalPoor: (d: MilkyWayData) => medianText(d.imigCells, Number.NEGATIVE_INFINITY, -0.1),
  cellsMetalSolar: (d: MilkyWayData) => medianText(d.imigCells, -0.1, 0.1),
  cellsMetalRich: (d: MilkyWayData) => medianText(d.imigCells, 0.1, Number.POSITIVE_INFINITY),
  cellsMatches: (d: MilkyWayData) => count(d.imigCells.nMatches),
  boxA: (d: MilkyWayData) => boxText(targeted(d, 0).box),
  boxB: (d: MilkyWayData) => boxText(targeted(d, 1).box),
  boxMatchesA: (d: MilkyWayData) => count(targeted(d, 0).cells.nMatches),
  boxMatchesB: (d: MilkyWayData) => count(targeted(d, 1).cells.nMatches),
  boxPoorA: (d: MilkyWayData) => medianText(targeted(d, 0).cells, Number.NEGATIVE_INFINITY, -0.1),
  boxPoorB: (d: MilkyWayData) => medianText(targeted(d, 1).cells, Number.NEGATIVE_INFINITY, -0.1),
  boxRichA: (d: MilkyWayData) => medianText(targeted(d, 0).cells, 0.1, Number.POSITIVE_INFINITY),
  boxRichB: (d: MilkyWayData) => medianText(targeted(d, 1).cells, 0.1, Number.POSITIVE_INFINITY),
  boxTrendA: (d: MilkyWayData) => trendText(targeted(d, 0).cells.trends.counted.metallicity),
  boxTrendB: (d: MilkyWayData) => trendText(targeted(d, 1).cells.trends.counted.metallicity),
  slopeCountsApogee: (d: MilkyWayData) => trendText(d.imigCells.slopes.counts),
  slopeExpApogee: (d: MilkyWayData) => trendText(d.imigCells.slopes.countsExponential),
  slopeSunApogee: (d: MilkyWayData) => trendText(d.imigCells.slopes.imigSun),
  slopeTracerApogee: (d: MilkyWayData) => trendText(d.imigCells.slopes.tracer),
  slopeSigmaApogee: (d: MilkyWayData) => trendText(d.imigCells.slopes.sigmaZ),
  slopeCountsLamost: (d: MilkyWayData) => trendText(d.imigCellsLamost.slopes.counts),
  slopeExpLamost: (d: MilkyWayData) => trendText(d.imigCellsLamost.slopes.countsExponential),
  slopeSunLamost: (d: MilkyWayData) => trendText(d.imigCellsLamost.slopes.imigSun),
  slopeTracerLamost: (d: MilkyWayData) => trendText(d.imigCellsLamost.slopes.tracer),
  estimatorRatios: (d: MilkyWayData) =>
    between(extent(d.estimators.map((e) => e.windowed / e.exponential))),
  estimatorSigmas: (d: MilkyWayData) => between(extent(d.estimators.map((e) => e.sigmaZ)), 0),
  dispersionCounts: (d: MilkyWayData) => dispersionText(d.imigCells.dispersion.countsExponential),
  dispersionImig: (d: MilkyWayData) => dispersionText(d.imigCells.dispersion.imigExponential),
  dispersionSun: (d: MilkyWayData) => dispersionText(d.imigCells.dispersion.imigSun),
  tracerGrowth: (d: MilkyWayData) => `$\\sigma_z^{${tracerGrowth(d.estimators).slope.toFixed(2)}}$`,
  estimatorWhole: (d: MilkyWayData) =>
    between(extent(d.estimators.map((e) => e.exponentialWhole / e.exponential))),
  gradientRange: (d: MilkyWayData) =>
    between(extent(d.surveys.flatMap((s) => s.bins.map((b) => b.gradient)))),
  distmassTracerOld: (d: MilkyWayData) =>
    between(
      extent(
        survey(d, 'apogee_distmass')
          .bins.filter((b) => b.ageLo >= 4)
          .map((b) => b.ratioTracer.value)
      )
    ),
  biasApogee15: (d: MilkyWayData) => biasText(d, 'apogee', 0.15),
  biasApogee20: (d: MilkyWayData) => biasText(d, 'apogee', 0.2),
  biasApogee40: (d: MilkyWayData) => biasText(d, 'apogee', 0.4),
  modelGeometry: (d: MilkyWayData) => cellRange(d, (c) => c.theirs.window / c.theirs.sun.value),
  modelSlices: (d: MilkyWayData) => cellRange(d, (c) => c.theirs.counted.value / c.theirs.window),
  theirExponential: (d: MilkyWayData) =>
    cellRange(d, (c) => c.theirs.exponential.value / c.theirs.sun.value),
  shapePoorMidplane: (d: MilkyWayData) =>
    fixed(medianShape(d, Number.NEGATIVE_INFINITY, -0.1)[0] ?? Number.NaN),
  shapeRichMidplane: (d: MilkyWayData) =>
    fixed(medianShape(d, 0.1, Number.POSITIVE_INFINITY)[0] ?? Number.NaN),
  shapeRichPeak: (d: MilkyWayData) =>
    fixed(Math.max(...medianShape(d, 0.1, Number.POSITIVE_INFINITY))),
  lamostCells: (d: MilkyWayData) => String(d.imigCellsLamost.nMeasured),
  lamostCellMatches: (d: MilkyWayData) => count(d.imigCellsLamost.nMatches),
  cellsStars: (d: MilkyWayData) => {
    const measured = d.imigCells.cells.filter((c) => isMeasured(d.imigCells, c))
    const sum = (cells: readonly { readonly nStars: number }[]) =>
      cells.reduce((total, c) => total + c.nStars, 0)
    return `${count(sum(measured))} of their ${count(sum(d.imigCells.cells))} stars`
  }
} satisfies Record<string, (d: MilkyWayData) => string>

export const JR_METRICS = {
  slopePalicio: (d: JrRelationData) => lineText(d.linePalicio),
  slopeScaled: (d: JrRelationData) => lineText(d.lineScaled),
  palicioSlope: (d: JrRelationData) => fixed(d.palicioSlope, 1),
  jrRange: (d: JrRelationData) => between(extent(d.points.map((p) => p.jrW)), 1),
  hzPalicioRange: (d: JrRelationData) => between(extent(d.points.map((p) => p.hzPalicio))),
  hzScaledRange: (d: JrRelationData) => between(extent(d.points.map((p) => p.hzScaled))),
  largestShifts: (d: JrRelationData) =>
    [...d.points]
      .sort((a, b) => b.hzPalicio / b.hzScaled - a.hzPalicio / a.hzScaled)
      .slice(0, 2)
      .map((p) => `${p.halo} (${fixed(p.hzPalicio / p.hzScaled)})`)
      .join(' and '),
  scaledDeviation: (d: JrRelationData) =>
    fixed(Math.abs(d.lineScaled.slope - d.palicioSlope) / d.lineScaled.slopeErr, 1),
  palicioDeviation: (d: JrRelationData) =>
    fixed(Math.abs(d.linePalicio.slope - d.palicioSlope) / d.linePalicio.slopeErr, 1),
  slopePalicioMixture: (d: JrRelationData) => lineText(d.linePalicioMixture),
  slopeScaledMixture: (d: JrRelationData) => lineText(d.lineScaledMixture),
  scaledMixtureShift: (d: JrRelationData) =>
    fixed(Math.abs(d.lineScaledMixture.slope - d.lineScaled.slope) / d.lineScaled.slopeErr, 1),
  scaledMixtureDeviation: (d: JrRelationData) =>
    fixed(Math.abs(d.lineScaledMixture.slope - d.palicioSlope) / d.lineScaledMixture.slopeErr, 1),
  palicioIntercept: (d: JrRelationData) => fixed(d.palicioIntercept, 1),
  leaveOneOutRange: (d: JrRelationData) =>
    between(extent(d.leaveOneOut.map((left) => left.lines.lineScaled.slope)), 0),
  influentialHalo: (d: JrRelationData) => mostInfluential(d).halo,
  influentialJr: (d: JrRelationData) => {
    const halo = mostInfluential(d).halo
    return fixed(d.points.find((p) => p.halo === halo)?.jrW ?? Number.NaN, 1)
  },
  otherJrMax: (d: JrRelationData) => {
    const halo = mostInfluential(d).halo
    return fixed(Math.max(...d.points.filter((p) => p.halo !== halo).map((p) => p.jrW)), 1)
  },
  withoutInfluentialScaled: (d: JrRelationData) => lineText(mostInfluential(d).lines.lineScaled),
  withoutInfluentialMixture: (d: JrRelationData) =>
    lineText(mostInfluential(d).lines.lineScaledMixture),
  leaveOneOutMixtureShift: (d: JrRelationData) =>
    fixed(
      Math.max(
        ...d.leaveOneOut.map(
          ({ lines }) =>
            Math.abs(lines.lineScaledMixture.slope - lines.lineScaled.slope) /
            lines.lineScaled.slopeErr
        )
      ),
      1
    )
} satisfies Record<string, (d: JrRelationData) => string>

/** The range of *survey* that starts at `lo` Gyr. */
function surveyRange(survey: MwSurvey, lo: number): MwRange {
  const found = survey.ranges.find((r) => r.lo === lo)
  if (found === undefined) {
    throw new Error(`no range from ${lo} Gyr in ${survey.name}`)
  }
  return found
}

/** Every survey of every realization of the ages drawn again. */
function redrawn(d: MwChecks): readonly MwSurvey[] {
  return d.ageErrors.flat()
}

/** h_Z of the bright over the faint half of the tracers in every bin, and its distance from 1 in errors. */
function magnitudeRatios(
  d: MwChecks,
  keep: (survey: string) => boolean = () => true
): readonly { readonly ratio: number; readonly sigmas: number }[] {
  return d.magnitude
    .filter((split) => keep(split.survey))
    .flatMap((split) =>
      split.bright.bins.flatMap((b, i) => {
        const f = split.faint.bins[i]
        const ratio = b.measured.value / f.measured.value
        const sigmas =
          Math.abs(b.measured.value - f.measured.value) / Math.hypot(b.measured.err, f.measured.err)
        return Number.isFinite(ratio) && Number.isFinite(sigmas) ? [{ ratio, sigmas }] : []
      })
    )
}

export const CHECK_METRICS = {
  ageRealizations: (d: MwChecks) => String(d.ageErrors.length),
  checkBootstrap: (d: MwChecks) => String(d.nBootstrap),
  ageTracerMedians: (d: MwChecks) =>
    between(extent(redrawn(d).map((s) => median(s.bins.map((b) => b.ratioTracer.value))))),
  ageTracerTrends0: (d: MwChecks) =>
    between(extent(redrawn(d).map((s) => surveyRange(s, 0).tracer.value)), 3),
  ageTracerTrendMax0: (d: MwChecks) =>
    fixed(
      Math.max(
        ...redrawn(d).map((s) => {
          const t = surveyRange(s, 0).tracer
          return Math.abs(t.value) / t.err
        })
      ),
      1
    ),
  ageEta0: (d: MwChecks) => between(extent(redrawn(d).map((s) => surveyRange(s, 0).eta.value))),
  magnitudeRatioRange: (d: MwChecks) => between(extent(magnitudeRatios(d).map((r) => r.ratio))),
  magnitudeWithin2: (d: MwChecks) => {
    const all = magnitudeRatios(d)
    return `${all.filter((r) => r.sigmas <= 2).length} of ${all.length}`
  },
  magnitudeLamostWithin2: (d: MwChecks) => {
    const all = magnitudeRatios(d, (s) => s === 'lamost')
    return `${all.filter((r) => r.sigmas <= 2).length} of ${all.length}`
  },
  magnitudeApogeeWithin2: (d: MwChecks) => {
    const all = magnitudeRatios(d, (s) => s !== 'lamost')
    return `${all.filter((r) => r.sigmas <= 2).length} of ${all.length}`
  },
  magnitudeMaxSigmas: (d: MwChecks) =>
    fixed(Math.max(...magnitudeRatios(d).map((r) => r.sigmas)), 1),
  magnitudeSplits: (d: MwChecks) => between(extent(d.magnitude.map((m) => m.gSplit)), 1)
} satisfies Record<string, (d: MwChecks) => string>
