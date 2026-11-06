#!/usr/bin/env python3
"""Atari image converter GUI launcher.

Mirrors ``convert.py`` for the CLI: the root script only makes ``src`` importable and delegates to
:func:`converter_gui.app.main`, so no Qt initialisation is duplicated here.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from converter_gui.app import main

if __name__ == "__main__":
    raise SystemExit(main())
