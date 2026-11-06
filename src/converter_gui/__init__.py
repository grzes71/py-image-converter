"""PySide6 desktop application for the Atari image converter.

The GUI is an **optional** part of the project: the package itself imports no Qt, so tools can
inspect it (and ``converter`` stays usable) without PySide6 installed. The Qt imports live in
:mod:`converter_gui.app` and :mod:`converter_gui.main_window`; PySide6 is the ``gui`` extra.

Start it with ``python -m converter_gui``, ``python convert_gui.py`` or the ``atari-convert-gui``
console script.
"""
