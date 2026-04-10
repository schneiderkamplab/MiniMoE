"""Brainsurgery entrypoint with tokcleanse custom transforms registered."""

from __future__ import annotations

import sys

from brainsurgery import main as brainsurgery_main

from . import _brainsurgery_transforms  # noqa: F401


def main(argv: list[str] | None = None) -> None:
    """Run brainsurgery with tokcleanse-specific transforms available."""

    raw_args = list(sys.argv[1:] if argv is None else argv)
    brainsurgery_main(["cli", *raw_args])


if __name__ == "__main__":
    main()
