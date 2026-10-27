# jr-hz-mono-age

**Mono-age disk populations and the age dependence of the spiral $J_R$-$h_Z$ relation.**

Palicio et al. (2025) found a linear relation between the maximum radial action of
spiral-arm stars and the vertical scale height $h_Z$ across 23 simulated disks. The
companion project, [`jr-hz-relation`](https://github.com/caverac/jr-hz-relation), derives
its slope for a whole disk from the self-gravitating isothermal sheet. This project asks
how the slope $a = J_{R,\rm med}/h_Z$ of populations of a single age behaves, using the
Auriga simulations and Milky Way survey data.

Documentation: <https://caverac.github.io/jr-hz-mono-age/>

## Table of Contents

- [Packages](#packages)
- [Getting Started](#getting-started)
- [Command-line interface](#command-line-interface)
- [Acknowledgments](#acknowledgments)
- [License](#license)

## Packages

| Package                               | Description                                                                                                                   |
| ------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| [`experiments`](packages/experiments) | Core Python engine and CLI                                                                                                    |
| [`docs`](packages/docs)               | Project documentation (Docusaurus), deployed at [caverac.github.io/jr-hz-mono-age](https://caverac.github.io/jr-hz-mono-age/) |
| [`pre-print`](packages/pre-print)     | LaTeX source (AASTeX v7) for the accompanying paper                                                                           |

## Getting Started

This project uses [mise](https://mise.jdx.dev/) to manage tool versions (Python 3.14,
Node 25, uv; see `.mise.toml`).

```bash
mise install               # tool versions
uv sync --group auriga     # Python engine, dev tools and AGAMA (pinned commit, ~18 min the first time)
yarn install               # JS/TS dependencies (docs site)
```

The Auriga snapshots (several GB per halo) can be downloaded through Globus
with `uv run experiments auriga download`
and analyzed with `uv run experiments auriga analyze`, Results and all scripts are persisted
and version controlled so you can reproduce the analysis. [`packages/experiments`](packages/experiments/README.md) lists the
steps.

## Command-line interface

```bash
uv run experiments --help          # list all commands
uv run experiments figures         # draw every figure from the cached data
uv run pytest                      # run the tests + 100% coverage gate
make -C packages/pre-print         # sync figures and build the pre-print PDF
yarn workspace @jr-hz-mono-age/docs start
```

## Acknowledgments

The simulated disks come from the Auriga Project

> We have used simulations from the Auriga Project public data release (Grand et al. 2024) available at https://wwwmpa.mpa-garching.mpg.de/auriga/data.html

- Grand, R. J. J., G&oacute;mez, F. A., Marinacci, F., et al. 2017, MNRAS, 467, 179,
  [doi:10.1093/mnras/stx071](https://doi.org/10.1093/mnras/stx071)
- Grand, R. J. J., Fragkoudi, F., G&oacute;mez, F. A., et al. 2024, MNRAS, 532, 1814,
  [doi:10.1093/mnras/stae1598](https://doi.org/10.1093/mnras/stae1598)

## License

This project is licensed under the [MIT License](LICENSE).
