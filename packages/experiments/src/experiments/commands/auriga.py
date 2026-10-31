"""``experiments auriga``: download the Auriga snapshots, check the local copies and analyze them.

The release is served only through Globus. ``download`` submits one Globus transfer per
halo into the local root, ``status`` follows the tasks and ``check`` reads the copies.
``mock-check`` runs the analysis on a mock whose answer is known, and ``analyze``
measures the halos. ``render-data`` bins their star particles for the images, and
``plot`` and ``render`` draw the figures from the cached data. ``jr-map`` measures a
proxy for the J_R,w of Palicio et al. (2025) from the map of the median J_R, and
``jr-relation`` fits it against the whole-disk h_Z across the halos. ``equilibrium``
measures the mean vertical motions of every population, ``tracer-sum`` predicts the
whole-disk h_Z as a sum of isothermal tracers, and ``concepts`` records what the Concepts pages of
the documentation draw::

    experiments auriga download 2 3 16 19 23 24 27
    caffeinate -i experiments auriga status 2 3 16 19 23 24 27 --wait
    experiments auriga check 2 3 16 19 23 24 27
    experiments auriga mock-check
    caffeinate -i experiments auriga analyze 2 3 16 19 23 24 27
    experiments auriga render-data 2 3 16 19 23 24 27
    experiments auriga plot 2 3 16 19 23 24 27
    experiments auriga render 2 3 16 19 23 24 27
    experiments auriga jr-map 2 3 16 19 23 24 27
    experiments auriga jr-relation 2 3 16 19 23 24 27
    experiments auriga equilibrium 2 3 16 19 23 24 27
    experiments auriga tracer-sum 2 3 16 19 23 24 27
    experiments auriga concepts 2 3 16 19 23 24 27
"""

from __future__ import annotations

import click

from experiments.commands.auriga_analyze import auriga_analyze
from experiments.commands.auriga_check import auriga_check
from experiments.commands.auriga_concepts import auriga_concepts
from experiments.commands.auriga_download import auriga_download
from experiments.commands.auriga_equilibrium import auriga_equilibrium
from experiments.commands.auriga_jr_map import auriga_jr_map
from experiments.commands.auriga_jr_relation import auriga_jr_relation
from experiments.commands.auriga_mock_check import auriga_mock_check
from experiments.commands.auriga_plot import auriga_plot
from experiments.commands.auriga_render import auriga_render
from experiments.commands.auriga_render_data import auriga_render_data
from experiments.commands.auriga_status import auriga_status
from experiments.commands.auriga_tracer_sum import auriga_tracer_sum


@click.group("auriga")
def auriga() -> None:
    """Download Auriga snapshots through Globus, check the local copies, analyze them and draw them."""


auriga.add_command(auriga_download)
auriga.add_command(auriga_status)
auriga.add_command(auriga_check)
auriga.add_command(auriga_mock_check)
auriga.add_command(auriga_analyze)
auriga.add_command(auriga_render_data)
auriga.add_command(auriga_plot)
auriga.add_command(auriga_render)
auriga.add_command(auriga_jr_map)
auriga.add_command(auriga_jr_relation)
auriga.add_command(auriga_equilibrium)
auriga.add_command(auriga_tracer_sum)
auriga.add_command(auriga_concepts)
