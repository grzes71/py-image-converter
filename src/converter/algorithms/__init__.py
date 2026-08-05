"""Dithering algorithm implementations."""

from converter.algorithms.atkinson import Atkinson
from converter.algorithms.bayer2 import Bayer2
from converter.algorithms.bayer2v2 import Bayer2v2
from converter.algorithms.bayer2x4 import Bayer2x4
from converter.algorithms.bayer4 import Bayer4
from converter.algorithms.bayer8 import Bayer8
from converter.algorithms.burkes import Burkes
from converter.algorithms.diagonal import Diagonal
from converter.algorithms.false_floyd import FalseFloydSteinberg
from converter.algorithms.floyd import FloydSteinberg
from converter.algorithms.jarvis import Jarvis
from converter.algorithms.non_rectangular8 import NonRectangular8
from converter.algorithms.sierra import Sierra
from converter.algorithms.sierra_lite import SierraLite
from converter.algorithms.stucki import Stucki
from converter.algorithms.two_row_sierra import TwoRowSierra

__all__ = [
    "Atkinson",
    "Bayer2",
    "Bayer2v2",
    "Bayer2x4",
    "Bayer4",
    "Bayer8",
    "Burkes",
    "Diagonal",
    "FalseFloydSteinberg",
    "FloydSteinberg",
    "Jarvis",
    "NonRectangular8",
    "Sierra",
    "SierraLite",
    "Stucki",
    "TwoRowSierra",
]
