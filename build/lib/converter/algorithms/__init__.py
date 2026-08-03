"""Dithering algorithm implementations."""

from converter.algorithms.atkinson import Atkinson
from converter.algorithms.bayer2 import Bayer2
from converter.algorithms.bayer4 import Bayer4
from converter.algorithms.bayer8 import Bayer8
from converter.algorithms.burkes import Burkes
from converter.algorithms.floyd import FloydSteinberg
from converter.algorithms.jarvis import Jarvis
from converter.algorithms.sierra import Sierra
from converter.algorithms.sierra_lite import SierraLite
from converter.algorithms.stucki import Stucki

__all__ = [
    "Atkinson",
    "Bayer2",
    "Bayer4",
    "Bayer8",
    "Burkes",
    "FloydSteinberg",
    "Jarvis",
    "Sierra",
    "SierraLite",
    "Stucki",
]
