"""Tests for the click command-line interface."""

from click.testing import CliRunner, Result

from experiments import __version__
from experiments.cli import AURIGA_DATA_DIR, AURIGA_ROOT, DATA_DIR, FIGURES_DIR, MOCK_ROOT, REPO_ROOT, main


def _run(*args: str) -> Result:
    """Invoke the CLI with the given arguments and return the result."""
    return CliRunner().invoke(main, list(args))


def test_help_lists_commands() -> None:
    """The group ``--help`` succeeds and lists the ``paths`` command."""
    result = _run("--help")
    assert result.exit_code == 0
    assert "paths" in result.output


def test_version_option() -> None:
    """``--version`` prints the package version."""
    result = _run("--version")
    assert result.exit_code == 0
    assert __version__ in result.output


def test_paths_point_into_the_repository() -> None:
    """``paths`` prints every location, all inside the repository root."""
    result = _run("paths")
    assert result.exit_code == 0
    for path in (AURIGA_ROOT, MOCK_ROOT, DATA_DIR, AURIGA_DATA_DIR, FIGURES_DIR):
        assert str(path) in result.output
        assert path.is_relative_to(REPO_ROOT)
    assert (REPO_ROOT / "pyproject.toml").is_file()
