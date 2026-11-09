"""Run the Capicove speech setup CLI from a source checkout."""

import sys

from capicove.cli.app import main


if __name__ == "__main__":
    raise SystemExit(main(["audio", "setup-speech", *sys.argv[1:]]))
