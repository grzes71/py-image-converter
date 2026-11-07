"""Reading the colour bytes out of a generated XEX (shared by the export tests).

The init routine at ``$3800`` is a fixed sequence of ``LDA #value / STA address`` writes, so the
``COLOR0..COLOR4`` shadows can be read back exactly the way the Atari sees them - which is the only
honest way to test what an export really writes.
"""

from __future__ import annotations

from pathlib import Path

INIT_ADDR = 0x3800
COLOR_BASE = 0x02C4
COLOR_END = 0x02C8


def segments(path: Path) -> list[tuple[int, bytes]]:
    """``(start address, payload)`` of every XEX segment, in file order."""
    data = Path(path).read_bytes()
    assert data[:2] == b"\xff\xff", "XEX magic"
    position = 2
    found: list[tuple[int, bytes]] = []
    while position < len(data):
        start = data[position] | (data[position + 1] << 8)
        end = data[position + 2] | (data[position + 3] << 8)
        position += 4
        size = end - start + 1
        found.append((start, data[position : position + size]))
        position += size
    return found


def colour_shadows(path: Path) -> dict[int, int]:
    """The ``COLOR0..COLOR4`` values the init routine stores, keyed by shadow address."""
    init = next(payload for start, payload in segments(path) if start == INIT_ADDR)

    assert init[0:5] == bytes([0xA9, 0x22, 0x8D, 0x29, 0x02]), "SDMCTL is set first"
    position = 15  # SDMCTL (5) + SDLSTL (5) + SDLSTH (5)
    shadows: dict[int, int] = {}
    while position + 4 < len(init) and init[position] == 0xA9:
        assert init[position + 2] == 0x8D, "the value is stored with STA abs"
        address = init[position + 3] | (init[position + 4] << 8)
        shadows[address] = init[position + 1]
        position += 5

    assert init[position] == 0x4C, "the routine ends with JMP back to itself"
    assert position + 3 == len(init)
    return shadows


def colour_registers(path: Path) -> list[int]:
    """The five shadow bytes in ``COLOR0..COLOR4`` order."""
    shadows = colour_shadows(path)
    return [shadows[address] for address in range(COLOR_BASE, COLOR_END + 1)]
