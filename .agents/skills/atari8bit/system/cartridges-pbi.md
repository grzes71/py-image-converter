---
name: atari8bit-cartridges-pbi
description: >-
  Atari 8-bit cartridge windows, cartridge banking, PBI expansion devices,
  internal devices, and reverse-engineering notes.
---

# Cartridges, PBI, and Internal Devices

> **When to load:** The task involves ROM carts, banked carts, PBI devices, IDE/SIDE-like hardware, internal clocks, or device-space reversing.
> **Source note:** This file is self-contained.

## Cartridge Basics

Atari 8-bit cartridges map ROM into fixed CPU address windows. Reverse them as banked memory devices, not as ordinary XEX files.

Common windows:

| Window | Signal / use |
|---|---|
| `$8000-$9FFF` | Lower 8K cartridge window, selected by `S4`; presence signaled by `RD4` |
| `$A000-$BFFF` | Upper 8K cartridge window, selected by `S5`; presence signaled by `RD5` |
| `$BFFA-$BFFF` | cartridge vectors/signature area in many formats |
| `$D500-$D5FF` | `CCTL` cartridge control area, commonly used for bank switching or cart hardware registers |

On the 800, the left cartridge slot can map the left window, right window,
and CCTL; the right slot can map the right window and CCTL. Other computer
models expose a single cartridge slot. The cartridge bus can support reads
and writes if the cartridge hardware implements them.

### CCTL false-read hazard

Cartridges often put bank-switch registers in `$D500-$D5FF`. These registers
may trigger on reads as well as writes. False reads from indexed addressing
modes and DMA can therefore switch banks unexpectedly:

- `LDA $D5FF,X` with `X=$08` targets `$D607`, but the CPU can perform a
  false read at `$D507`, tripping a cartridge bank register.
- ANTIC DMA from a display list or playfield address in `$D5xx` can also hit
  CCTL and switch banks. An accidentally active display-list byte pattern
  such as `$D5 $D5` can make ANTIC read `$D5D5`.

Avoid indexed reads/writes that cross through CCTL/PBI control pages, and
keep active display lists out of hardware control ranges.

### Reset and startup behavior

The cartridge port does not expose the computer reset signal in the normal
way. A cartridge may have a reliable power-on reset circuit, no reset circuit,
or delayed reset behavior. Banked cartridges should contain startup code in
every possible power-up bank, or every bank should vector quickly to the real
startup bank. Warm reset may not reset cartridge banking hardware.

Reversing workflow:

1. Identify ROM size and visible initial bank.
2. Locate cartridge vectors/signature bytes.
3. Watch writes to `$D5xx` and any documented control range.
4. Build a bank-switch table before linear disassembly.
5. Treat code reached after a bank write as a different address-space state.

## Banked Cartridge Rules

- A CPU address is not enough to identify code; record `(bank, address)`.
- Bank registers can be write-only; infer current bank from the last write.
- Interrupt handlers must live in always-visible ROM/RAM or restore the expected bank before executing.
- Self-modifying code cannot modify ROM, but can copy routines to RAM and patch the RAM copy.
- Reads or writes to `$D5xx` may have side effects even if the value is ignored; log the address low nibble and address bits, not only the data bus value.
- Some carts turn themselves off by clearing `RD4`/`RD5`; after an off command, code may continue from RAM or OS vectors.

Common cartridge families useful for reversing:

| Family | Bank-control rule |
|---|---|
| OSS 16K | 4K banks: fixed bank in `$B000-$BFFF`, selectable bank in `$A000-$AFFF`; control often depends on address bits A0/A3. |
| SpartaDOS X / Diamond / Express | 8 x 8K banks in `$A000-$BFFF`; base ranges differ, `base+0..7` selects bank and `base+8..F` turns cart off. |
| XEGS | Last 8K bank fixed at `$A000-$BFFF`; banks 0..N-1 switch into `$8000-$9FFF` by writing bank number to `$D500-$D5FF`. |
| R-Time 8 | Pass-through cartridge with RTC hardware; reads are often enough for clock access. |
| MaxFlash 1Mbit | 128K flash, 8K window at `$A000-$BFFF`; `$D500-$D51F` selects bank with low address bits and disables with bit 4. |
| MaxFlash 8Mbit | 1MB flash, 8K window at `$A000-$BFFF`; `$D500-$D5FF` selects bank with low 7 address bits and disables with bit 7. |
| MaxFlash 1Mbit + MyIDE | MaxFlash-style banking moved to `$D520-$D53F`; CompactFlash ATA registers at `$D500-$D507`. |
| MyIDE-II | Flash/RAM plus CompactFlash. Independent 8K windows at `$8000-$9FFF` and `$A000-$BFFF`; CF registers at `$D500-$D507`; keyhole window at `$D580-$D5FF`. |

CompactFlash interfaces exposed through cartridge hardware are commonly
8-bit ATA register windows. If the CF device is unpowered or held in reset,
read data may be undriven or status-only depending on the register range.

## PBI Overview

The Parallel Bus Interface exposes expansion hardware on XL-class systems. PBI devices can add ROM handlers, disk interfaces, memory, clocks, or other peripherals.

PBI reserves `$D1xx`, `$D6xx`, and `$D7xx` for the active device. `$D1FF` is
conventionally both the PBI device select register and the PBI IRQ status
register:

- As `PDVS`, writing one bit selects the corresponding device; writing `$00`
  deselects devices. Only one bit should be selected at a time.
- As `PDVI`, a set bit reports that the corresponding device is requesting
  an interrupt.
- PBI devices are not forced by the base computer to implement `$D1FF`
  exactly; some partially decode the address or overload bits with
  device-specific control.

An active PBI device can overlay firmware ROM into `$D800-$DFFF` by asserting
Math Pack Disable (`MPD`). PBI can overlay RAM but not ordinary I/O or
cartridge address space. On a 130XE, extended RAM also uses `/CASINH`, which
can interact with PBI overlay behavior. Hardware expansions that implement
extended RAM outside the motherboard memory system may behave differently.

Key hardware behavior:

| Signal | Meaning for software/reversing |
|---|---|
| `EXTENB` | External decoder enable; high when address is in a PBI-allowed device range. |
| `EXTSEL` | Selected device pulls low when it owns the addressed range. |
| `MPD` | Math-pack disable; lets a device provide handler ROM in `$D800-$DFFF`. |
| `RDY` | A slow device can extend a CPU bus cycle by pulling it low. |
| `IRQ` | PBI devices can interrupt the CPU; preserve/identify IRQ vectors when tracing. |
| `AUDIO IN` | Expansion audio can mix into the computer audio path. |

600XL and 800XL both expose PBI, but 600XL has +5V lines used by the 1064 memory expansion. Most XE systems expose the reduced ECI connector plus the cartridge connector instead of a full PBI; ECI carries selected PBI signals such as `EXTSEL`, `IRQ`, `HALT`, `D1XX`, `MPD`, `REF`, and audio in.

Agent rules:

- If code scans device IDs or handler tables, treat it as expansion discovery.
- If code patches CIO/SIO behavior after reset, check for PBI-resident handlers.
- If a program depends on PBI hardware, document fallback behavior or requirement.

### Common PBI/Internal Devices

| Device | Key software-visible facts |
|---|---|
| ICD MIO | PBI device with SCSI, printer, RS-232, and RAM. Occupies `$D100-$D1FF` and `$D600-$D6FF` while active; uses registers around `$D1FC-$D1FF` for SCSI/printer data, status, control, and RAM bank selection. |
| Covox-style DAC | Usually a write-only 8-bit DAC mapped at ranges such as `$D600-$D6FF`, `$D700-$D7FF`, or `$D280-$D2FF`. Reads are normally not handled, so detection is unreliable. Stereo/4-channel versions select channels with low address bits. |
| Ultimate1MB | Internal multifunction expansion with extended RAM, selectable OS/BASIC/game ROM images, flash ROM, and SDX cartridge emulation. Flash command/query modes affect every visible mapping of the same flash chip, including OS/BASIC/cartridge windows. |
| VBXE | Often uses `$D6xx/$D7xx` register windows and can interact with PBI/ECI decode behavior. Use `exotics/vbxe.md` for register detail. |

Ultimate1MB caution: when flash writing/query mode is enabled, entering flash
autoselect or programming mode through one mapped window changes what every
other mapped flash window returns. This can make OS ROM vanish until the flash
returns to normal array-read mode; a reset while flash is in command mode may
require a power cycle.

## Internal Devices and Clocks

Internal or pseudo-internal devices can appear through OS vectors, PBI, or SIO commands.

Examples in this corpus:

| Device | Skill location |
|---|---|
| APE-Time RTC | `exotics/sophia-rapidus.md` |
| VBXE | `exotics/vbxe.md` |
| Sophia | `exotics/sophia-rapidus.md` |
| XEP80 and accessories | `system/input.md` |

## Reverse-Engineering Notes

- Hardware control ranges often decode incompletely and mirror across address ranges.
- Some devices require timing gaps after commands; do not collapse polling loops unless the hardware docs say it is safe.
- For emulation-targeted code, note whether Altirra supports the device and whether real hardware timing has been tested.
