# experiments

The analysis code and command-line interface behind the paper on mono-age disk
populations and the age dependence of the $J_R$-$h_Z$ relation. This is a
workspace-internal package, not a published PyPI distribution; it exists only to
derive every number and figure in `packages/pre-print/`.

- `cli.py`: the command line. `paths` prints where the engine reads the Auriga
  snapshots (`staging/auriga/`) and writes its data (`assets/data/`) and figures
  (`assets/figures/`).
- `auriga.py`, `globus_transfer.py`, `snapshot_check.py`: the layout of the Auriga
  release and the record of a download, the Globus transfers, and the integrity check
  of a local copy, behind the `auriga` commands.
- `cosmology.py`, `snapshot.py`, `potential.py`, `heights.py`, `populations.py`,
  `whole_disk.py`, `palicio.py`, `analysis.py`, `mock.py`: the analysis of a halo, from
  the snapshot to the mono-age populations, and the mock galaxy that checks it.
- `records.py`, `_plotting.py`, `figures.py`, `render.py`: reading the cached data back,
  the figure style, the figures of the analysis, and the rendered images of the disks.

AGAMA, which the analysis and its tests need, comes from the non-default `auriga`
dependency group, because it compiles GSL from source:

```bash
uv sync --group auriga       # engine, dev tools and AGAMA (~18 min the first time)
uv run pytest                # tests + 100% coverage gate
uv run experiments --help
```

## Downloading the Auriga snapshots

The Auriga release (Grand et al. 2024, MNRAS 532, 1814) is served only through Globus,
from the collection "The Auriga Simulations"
(`02a2dbb8-f64d-4440-bafe-44b60b964501`). A download is a Globus transfer from that
collection to a Globus Connect Personal (GCP) collection on this machine, which runs in
Docker (`docker/compose.yaml`) and can write to `staging/auriga/` only.
`experiments auriga download` submits one transfer per halo, of `snapdir_127/` and
`groups_127/` (z = 0 at Level 4), with checksums verified by Globus. It lists the
files of the release before submitting, and writes that listing, with the sizes and the
task ID, to `staging/auriga/level4/Original/halo_<N>/transfer.json`. `status` adds the
state of the task to the same file.

The release asks every publication that uses its data to cite Grand et al. (2017) and
Grand et al. (2024), and to carry its acknowledgment, which the
[root README](../../README.md#acknowledgments) quotes.

Steps 1 to 3 are done once per machine.

1. Register a native app in the Globus web app (Settings, Developers, "Register a
   thick client or script that will be installed and run by users on their devices"),
   and export its client UUID as `JR_HZ_GLOBUS_CLIENT_ID`, for example in `.envrc`.
2. Build the GCP image and register the endpoint under the name that `download`
   looks for. `-setup` prints a login URL and asks for the code the login shows:

   ```bash
   mkdir -p staging/auriga
   docker compose -f docker/compose.yaml build
   docker compose -f docker/compose.yaml run --rm globus \
       globusconnectpersonal -setup --name jr-hz-mono-age-auriga
   ```

3. Start the endpoint. Its status reads "Globus Online: connected":

   ```bash
   docker compose -f docker/compose.yaml up -d globus
   docker compose -f docker/compose.yaml exec globus globusconnectpersonal -status
   ```

4. Submit, follow and check the transfers. The first `download` opens a Globus login
   on the command line. `status --wait` polls every 5 minutes until every task has
   ended. Under `caffeinate -i` the Mac, and with it the endpoint, stays awake for as
   long as the command runs:

   ```bash
   uv run experiments auriga download 2 3 16 19 23 24 27
   caffeinate -i uv run experiments auriga status 2 3 16 19 23 24 27 --wait
   uv run experiments auriga check 2 3 16 19 23 24 27
   ```

`check` compares the local files with the listing in `transfer.json`, then opens every
HDF5 chunk and checks its row counts against the headers. It also reports the particles
that sit exactly on the center of the main subhalo, which the AGAMA potential fit must
leave out. The z = 0 snapshot and catalog of Au24 at Level 4 take 6.8 GB. The
`globus` service restarts with Docker Desktop until
`docker compose -f docker/compose.yaml stop globus`.

## Analyzing the halos

`experiments auriga analyze` reads each halo once, fits its potential, and measures its
mono-age populations and its whole disk. `experiments auriga mock-check` runs the same
analysis on a mock galaxy whose answer is known. Both print a report and write a JSON
record to `assets/data/auriga/`:

```bash
OMP_NUM_THREADS=4 caffeinate -i uv run experiments auriga mock-check
OMP_NUM_THREADS=4 caffeinate -i uv run experiments auriga analyze 2 3 16 19 23 24 27
```

The analysis of one halo runs in these steps:

1. `snapshot.py` reads the particles within 300 kpc of the main galaxy in physical
   units, drops wind particles, and rotates them into the frame of the angular momentum
   of the stars inside 10 kpc.
2. `potential.py` fits an axisymmetric AGAMA potential: a CylSpline for the gas and stars
   of the disk region and a Multipole for the rest, without the particle at the origin.
3. `populations.py` splits the star particles of $7.5 < R < 9$ kpc into 1 Gyr age bins.
   Per bin it measures $J_{R,\rm med}$, $\sigma_R$, $\sigma_z$, and $h_Z$ in the
   self-consistent window $k\,h_Z$ (`--k`, default 2) and in Palicio's fixed window.
   The isothermal sheet and the isothermal tracer predict $h_Z$ through the same
   windows, the tracer both in the fitted potential and in the snapshot's. Errors come
   from 300 bootstrap resamplings (`--n-bootstrap`).
4. `populations.py` fits the age exponents over 0-10, 2-10, 0-8 and 2-8 Gyr, and repeats
   the exponent of $a = J_{R,\rm med}/h_Z$ for $k = 1.5$, 2 and 3 and for fixed windows of
   0.5 to 2.5 kpc.
5. `whole_disk.py` measures all star particles in rings over Palicio's radial range.

Palicio's window of each halo, $Z_{\rm lim}$ and $L_{\rm kpc}$, comes from Palicio et al.
(2025, A&A 695, A193, Table 1) through `palicio.py`; `--z-lim` and `--l-kpc` replace it.
With `OMP_NUM_THREADS=4`, Au24 at Level 4 takes about 3 min and 3.7 GB of memory.

The mock (`mock.py`) samples an AGAMA disk and halo and five quasi-isothermal stellar
populations, and writes them as Arepo chunks, tilted, moving and wrapped across two faces
of the periodic box. `mock-check` compares the analysis with the truth: the spin axis,
the ages, $v_c$ and $\Delta\Phi(z)$ of the fit, and $h_Z$ and $J_{R,\rm med}$ per bin.

## Figures

As in [`jr-hz-relation`](https://github.com/caverac/jr-hz-relation), the expensive
commands write data to `assets/data/` and the plot commands draw from it, so that a
figure can be redrawn without the snapshots. A plot goes to `assets/figures/` as a PNG
for the documentation site and a PDF for the paper. A rendered image of a disk, which
only the documentation site shows, goes there as a JPEG, several times smaller than a
PNG for a smooth image. A figure is rewritten only when its pixels change.

```bash
uv run experiments auriga render-data 2 3 16 19 23 24 27   # maps and star samples, ~10 s per halo
uv run experiments auriga plot 2 3 16 19 23 24 27          # <stem>-mono-age, from the analyze records
uv run experiments auriga render 2 3 16 19 23 24 27        # <stem>-portrait.jpg, for the docs only
uv run experiments figures                                 # all of the above from whatever data is cached
```

`render-data` bins the star particles of each halo, in its disk frame, into face-on and
edge-on surface-density maps of five age groups (`<stem>-maps.npz`), and draws 80,000
star particles at random for the 3D view of the documentation (`<stem>-stars.bin` and
`<stem>-stars.json`). The maps, about 2.5 MB per halo, are ignored by git: they only feed
`render`, whose portraits are committed, so restyling the portraits means running
`render-data` on the snapshots first. In the rendered images a pixel is as bright as its stellar surface
density, on a logarithmic scale over three decades, and has the color of the mean age of
its stars. `render.py` describes the smoothing, the glow and the exposure curve.
