# jr-hz-mono-age

**Mono-age disk populations and the age dependence of the spiral $J_R$-$h_Z$ relation.**

Palicio et al. (2025) found a linear relation between the maximum radial action of
spiral-arm stars and the vertical scale height $h_Z$ across 23 simulated disks. The
companion project, [`jr-hz-relation`](https://github.com/caverac/jr-hz-relation), derives
its slope for a whole disk from the self-gravitating isothermal sheet. This project asks
how the slope $a = J_{R,\rm med}/h_Z$ of populations of a single age behaves, using the
Auriga simulations and Milky Way survey data.

## Table of Contents

- [Packages](#packages)
- [Getting Started](#getting-started)
- [Command-line interface](#command-line-interface)
- [License](#license)

## Packages

| Package                               | Description                                         |
| ------------------------------------- | --------------------------------------------------- |
| [`experiments`](packages/experiments) | Core Python engine and CLI                          |
| [`docs`](packages/docs)               | Project documentation (Docusaurus)                  |
| [`pre-print`](packages/pre-print)     | LaTeX source (AASTeX v7) for the accompanying paper |

## Getting Started

This project uses [mise](https://mise.jdx.dev/) to manage tool versions (Python 3.14,
Node 25, uv; see `.mise.toml`).

```bash
mise install               # tool versions
uv sync                    # Python engine and dev tools (uv workspace)
uv sync --group auriga     # adds AGAMA, compiled from a pinned commit (~18 min)
yarn install               # JS/TS dependencies (docs site)
```

The Auriga snapshots (several GB per halo) are read from `staging/auriga/`, which git
ignores. They are downloaded through Globus with `uv run experiments auriga download`;
[`packages/experiments`](packages/experiments/README.md) lists the steps.

## Command-line interface

```bash
uv run experiments --help          # list all commands
uv run pytest                      # run the tests + 100% coverage gate
make -C packages/pre-print         # sync figures and build the pre-print PDF
yarn workspace @jr-hz-mono-age/docs start
```

## License

This project is licensed under the [MIT License](LICENSE).
