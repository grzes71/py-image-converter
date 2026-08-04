#!/usr/bin/env python3
"""Atari image converter CLI."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from converter.main import main

if __name__ == "__main__":
    raise SystemExit(main())

