"""Entry point for ``python -m converter_gui``.

The actual start-up lives in :func:`converter_gui.app.main`, so the module runner and the console
script share one implementation.
"""

from __future__ import annotations

from converter_gui.app import main

if __name__ == "__main__":
    raise SystemExit(main())
