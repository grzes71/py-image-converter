"""Atari 8-bit image converter.

The public API is re-exported from the package, so client code - the GUI in particular - imports it
from ``converter`` instead of reaching into the individual modules::

    from converter import ConversionEngine, ConversionConfig, ConversionResult, convert_image

Every re-exported name depends on NumPy, which is already a hard requirement of the project, so the
imports are eager and no lazy loading is needed. Nothing here imports PySide6: the GUI is a separate
package that consumes this one.
"""

from importlib.metadata import PackageNotFoundError, version

from converter.engine import ConversionEngine
from converter.pipeline import convert_image
from converter.types import ConversionConfig, ConversionResult

try:
    __version__ = version("atari-image-converter")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "ConversionConfig",
    "ConversionEngine",
    "ConversionResult",
    "convert_image",
]
