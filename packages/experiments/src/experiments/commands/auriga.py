"""``experiments auriga``: download the Auriga snapshots and check the local copies.

The release is served only through Globus. ``download`` submits one Globus transfer per
halo into the local root, ``status`` follows the tasks and ``check`` reads the copies::

    experiments auriga download 2 3 16 19 23 27
    caffeinate -i experiments auriga status 2 3 16 19 23 27 --wait
    experiments auriga check 2 3 16 19 23 27
"""

from __future__ import annotations

import click

from experiments.commands.auriga_check import auriga_check
from experiments.commands.auriga_download import auriga_download
from experiments.commands.auriga_status import auriga_status


@click.group("auriga")
def auriga() -> None:
    """Download Auriga snapshots through Globus and check the local copies."""


auriga.add_command(auriga_download)
auriga.add_command(auriga_status)
auriga.add_command(auriga_check)
