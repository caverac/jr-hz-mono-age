"""``experiments auriga``: download the Auriga snapshots, check the local copies and analyze them.

The release is served only through Globus. ``download`` submits one Globus transfer per
halo into the local root, ``status`` follows the tasks and ``check`` reads the copies.
``mock-check`` runs the analysis on a mock whose answer is known, and ``analyze``
measures the halos::

    experiments auriga download 2 3 16 19 23 24 27
    caffeinate -i experiments auriga status 2 3 16 19 23 24 27 --wait
    experiments auriga check 2 3 16 19 23 24 27
    experiments auriga mock-check
    caffeinate -i experiments auriga analyze 2 3 16 19 23 24 27
"""

from __future__ import annotations

import click

from experiments.commands.auriga_analyze import auriga_analyze
from experiments.commands.auriga_check import auriga_check
from experiments.commands.auriga_download import auriga_download
from experiments.commands.auriga_mock_check import auriga_mock_check
from experiments.commands.auriga_status import auriga_status


@click.group("auriga")
def auriga() -> None:
    """Download Auriga snapshots through Globus, check the local copies and analyze them."""


auriga.add_command(auriga_download)
auriga.add_command(auriga_status)
auriga.add_command(auriga_check)
auriga.add_command(auriga_mock_check)
auriga.add_command(auriga_analyze)
