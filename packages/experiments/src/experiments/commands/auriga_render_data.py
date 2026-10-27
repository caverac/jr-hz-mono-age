"""``experiments auriga render-data``: bin the star particles of each halo for the images and the 3D view.

Each halo is read and rotated into its disk frame (:mod:`experiments.render`). The maps go
to ``<out-dir>/<stem>-maps.npz``, which git ignores: they only feed ``render``, whose
portraits are committed. The random sample of star particles, which the documentation
site loads, goes to ``<out-dir>/<stem>-stars.bin`` with its description in
``<stem>-stars.json``::

    experiments auriga render-data 2 3 16 19 23 24 27
"""

from __future__ import annotations

from pathlib import Path

import click
import numpy as np

from experiments._paths import AURIGA_DATA_DIR
from experiments.auriga import HaloSnapshot
from experiments.commands._auriga import snapshot_options, snapshots
from experiments.render import PIXELS, SAMPLE_SIZE, save_maps, star_maps, star_sample, write_sample
from experiments.snapshot import galaxy_frame, read_galaxy

SAMPLE_SEED = 20261027


def render_data(snapshot: HaloSnapshot, root: Path, pixels: int, sample: int, out_dir: Path) -> str:
    """Write the maps and the star sample of *snapshot* under *root* to *out_dir*; return the report."""
    try:
        galaxy = read_galaxy(snapshot, root)
    except FileNotFoundError as err:
        raise click.ClickException(f"{err}; download Au{snapshot.halo} with `experiments auriga download`") from err
    galaxy = galaxy.in_frame(*galaxy_frame(galaxy.stars))
    age = galaxy.header.cosmology.lookback_gyr(galaxy.stars.formation_a, galaxy.header.scale_factor)
    maps = save_maps(star_maps(galaxy.stars, age, pixels), out_dir / f"{snapshot.stem}-maps.npz")
    drawn = star_sample(galaxy.stars, age, sample, np.random.default_rng(SAMPLE_SEED))
    binary, meta = write_sample(drawn, f"Au{snapshot.halo}", out_dir / f"{snapshot.stem}-stars")
    return f"Au{snapshot.halo}: wrote {maps}, {binary} ({drawn.ages.size:,} star particles) and {meta}"


@click.command("render-data")
@snapshot_options
@click.option("--pixels", default=PIXELS, show_default=True, type=click.IntRange(min=16), help="Pixels across a map.")
@click.option(
    "--sample",
    default=SAMPLE_SIZE,
    show_default=True,
    type=click.IntRange(min=1),
    help="Star particles of the 3D view.",
)
@click.option(
    "--out-dir",
    default=AURIGA_DATA_DIR,
    show_default=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory of the maps and the samples.",
)
def auriga_render_data(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    halos: tuple[int, ...],
    level: int,
    suite: str,
    snapshot: int,
    root: Path,
    pixels: int,
    sample: int,
    out_dir: Path,
) -> None:
    """Bin the star particles of each of HALOS into age-group maps, and draw a sample for the 3D view."""
    for snap in snapshots(halos, level, suite, snapshot):
        click.echo(render_data(snap, root, pixels, sample, out_dir))
