"""Atari 8-bit image converter."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("atari-image-converter")
except PackageNotFoundError:
    __version__ = "0.0.0"
