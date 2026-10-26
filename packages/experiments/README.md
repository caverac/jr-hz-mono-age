# experiments

The analysis code and command-line interface behind the paper on mono-age disk
populations and the age dependence of the $J_R$-$h_Z$ relation. This is a
workspace-internal package, not a published PyPI distribution; it exists only to
derive every number and figure in `packages/pre-print/`.

- `cli.py`: the command line. `paths` prints where the engine reads the Auriga
  snapshots (`staging/auriga/`) and writes tables (`assets/data/`) and figures
  (`assets/figures/`).
- `auriga.py`, `globus_transfer.py`, `snapshot_check.py`: the layout of the Auriga
  release and the record of a download, the Globus transfers, and the integrity check
  of a local copy, behind the `auriga` commands.

The Auriga analysis (snapshot reading, the AGAMA potential fit and the scale-height
estimators) is not in the package yet. AGAMA comes from the non-default `auriga`
dependency group, because it compiles GSL from source:

```bash
uv sync                      # engine and dev tools
uv sync --group auriga       # adds AGAMA (~18 min the first time)
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
