"""``experiments auriga mock-check``: run the analysis on a mock snapshot whose answer is known.

The mock (:mod:`experiments.mock`) is written under ``--root`` as halo 0, analyzed exactly
as an Auriga halo with Au24's window from Palicio et al. (2025, Table 1), and compared
with its truth. The comparison is printed and written to ``<out-dir>/mock-check.json``::

    OMP_NUM_THREADS=4 experiments auriga mock-check
"""

from __future__ import annotations

from pathlib import Path

import click

from experiments._paths import AURIGA_DATA_DIR, MOCK_ROOT
from experiments.analysis import analyze_halo, prepare_halo
from experiments.auriga import HaloSnapshot
from experiments.commands._report import report_lines, write_json
from experiments.mock import MockCheck, MockSizes, build_mock, check_mock
from experiments.palicio import PALICIO_WINDOWS
from experiments.populations import K_MAIN, N_BOOTSTRAP, Method

MOCK_SNAPSHOT = HaloSnapshot(0)
COUNT = click.IntRange(min=1)


def check_lines(check: MockCheck) -> list[str]:
    """Return the printed comparison of the mock with its truth."""
    return [
        "mock check",
        f"  spin axis error {check.spin_error_deg:.4f} deg; largest age round-trip error {check.age_error_gyr:.2e} Gyr",
        f"  v_c(R_mid) {check.v_circ:.2f} km/s, true {check.v_circ_true:.2f} km/s; largest "
        f"|Delta Phi_fit / Delta Phi_true - 1| at 0.05 < z <= 1.5 kpc {check.delta_phi_error:.4f}",
        *(
            f"  {b.age_lo:.0f}-{b.age_hi:.0f} Gyr: h_Z {b.hz:.4f} vs {b.hz_true:.4f} ({b.hz / b.hz_true - 1.0:+.2e}), "
            f"J_R,med {b.jr_median:.2f} vs {b.jr_median_true:.2f} ({b.jr_median / b.jr_median_true - 1.0:+.3f}); "
            f"tracer h_Z {b.hz_tracer:.3f}"
            for b in check.bins
        ),
    ]


@click.command("mock-check")
@click.option(
    "--root",
    default=MOCK_ROOT,
    show_default=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Where the mock snapshot is written.",
)
@click.option(
    "--out-dir",
    default=AURIGA_DATA_DIR,
    show_default=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory of the JSON record.",
)
@click.option("--n-disk", default=MockSizes.n_disk, show_default=True, type=COUNT, help="Gas cells of the disk.")
@click.option("--n-halo", default=MockSizes.n_halo, show_default=True, type=COUNT, help="Dark matter particles.")
@click.option("--n-stars", default=MockSizes.n_stars, show_default=True, type=COUNT, help="Stars per population.")
@click.option(
    "--n-bootstrap", default=N_BOOTSTRAP, show_default=True, type=click.IntRange(min=2), help="Bootstrap resamplings."
)
def auriga_mock_check(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    root: Path, out_dir: Path, n_disk: int, n_halo: int, n_stars: int, n_bootstrap: int
) -> None:
    """Build a mock snapshot of a known galaxy, analyze it, and compare the analysis with the truth."""
    mock = build_mock(MOCK_SNAPSHOT, root, MockSizes(n_disk, n_halo, n_stars))
    prepared = prepare_halo(MOCK_SNAPSHOT, root)
    analysis = analyze_halo(prepared, Method(PALICIO_WINDOWS[24], K_MAIN, n_bootstrap))
    check = check_mock(mock, prepared, analysis)
    for line in (*report_lines(analysis), *check_lines(check)):
        click.echo(line)
    click.echo(f"  wrote {write_json(check, out_dir / 'mock-check.json')}")
