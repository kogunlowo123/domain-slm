"""Entry point for ``python -m dslm``."""

from __future__ import annotations

import sys

from dslm.cli import main

if __name__ == "__main__":
    sys.exit(main())
