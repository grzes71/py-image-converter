"""Product limits of the GUI (plan §3.4.1, §6.2).

These are GUI policy, not format limits: PNG stores up to 256 colours and the converter core accepts
them, but the GUI stops earlier to keep the interactive path responsive. The constants live here, in
the GUI package, so the CLI and the exporters keep the full capability.
"""

from __future__ import annotations

MAX_PNG_COLORS = 16
"""Colour cap for PNG-only exports.

PNG itself stores 256 colours; the cap is interactive speed. The palette stage grows with N (at 160x192
k-means takes ~49 ms at N=2, ~319 ms at N=16 and ~4.4 s at N=256), and the popularity/octree selectors
plus the dither loops are pure Python.
"""
