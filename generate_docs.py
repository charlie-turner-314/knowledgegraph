"""Utility script to generate project documentation using pdoc."""

from pathlib import Path

import pdoc


def main() -> None:
    """Render API documentation for the ``app`` package into ``docs/``."""

    output_dir = Path("docs")
    output_dir.mkdir(exist_ok=True)
    context = pdoc.Context()
    module = pdoc.Module("app", context=context)
    pdoc.link_inheritance(context)
    pdoc.pdoc("app", output_directory=str(output_dir))


if __name__ == "__main__":
    main()
