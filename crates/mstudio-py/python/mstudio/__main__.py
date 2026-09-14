"""`python -m mstudio [file]` / the `mstudio` console script: open the desktop app."""

from __future__ import annotations

import argparse
import sys

from . import __version__, run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mstudio", description="MStudio motion-capture marker viewer / editor")
    parser.add_argument("path", nargs="?", help=".trc, .c3d or a Pose2Sim / Sports2D JSON folder to open")
    parser.add_argument("--play", action="store_true", help="start playback immediately")
    parser.add_argument("--version", action="version", version=f"mstudio {__version__}")
    args = parser.parse_args(argv)
    run(args.path, play=args.play)
    return 0


if __name__ == "__main__":
    sys.exit(main())
