"""Enables ``python -m theory_validation <command>``."""

import sys

from theory_validation.cli import main

if __name__ == "__main__":
    sys.exit(main())
