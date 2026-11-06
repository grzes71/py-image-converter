"""Conversion workers.

``base`` holds the Qt-free :class:`~converter_gui.workers.base.ConversionBackend` protocol and is safe to
import without PySide6. ``conversion_worker`` holds the v1 ``QThreadBackend`` implementation and needs Qt,
so it is imported explicitly by the code that uses it.
"""
