"""Utility script to generate project documentation using pdoc."""

from pathlib import Path

import pdoc


def main() -> None:
    """Render API documentation for the ``app`` package into ``docs/``."""

    output_dir = Path("docs")
    output_dir.mkdir(exist_ok=True)
    pdoc.pdoc("app", output_directory=output_dir)


if __name__ == "__main__":
    main()
