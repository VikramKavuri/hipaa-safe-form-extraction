"""Command-line interface.

Usage::

    formextract run                      # process FORMEXTRACT_INPUT_FOLDER
    formextract run --input ./data/sample --output ./outputs/run.csv
    formextract info                     # show resolved configuration
"""

from __future__ import annotations

import logging
from pathlib import Path

import typer

from .config import get_settings
from .logging_utils import configure_logging
from .pipeline import run_batch

app = typer.Typer(
    add_completion=False, help="Privacy-first local document AI for fire-drill forms."
)


@app.command()
def run(
    input: Path | None = typer.Option(None, "--input", "-i", help="Folder of forms to process."),
    output: Path | None = typer.Option(None, "--output", "-o", help="Output CSV path."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable DEBUG logging."),
) -> None:
    """Process a folder of forms into one CSV row per form."""
    configure_logging(logging.DEBUG if verbose else logging.INFO)
    settings = get_settings()
    if input is not None:
        settings.input_folder = input
    if output is not None:
        settings.output_csv = output
    run_batch(settings)


@app.command()
def info() -> None:
    """Print the resolved configuration (useful for debugging env overrides)."""
    configure_logging()
    settings = get_settings()
    for key, value in settings.model_dump().items():
        typer.echo(f"{key:28s} = {value}")


if __name__ == "__main__":
    app()
