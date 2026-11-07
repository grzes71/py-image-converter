# XEX colour-register verification (plan task T0.3, defect B5)

> **Status: fixed.** The register-order fix (§11.1, ANTIC D/E) and the mode-F strategy (§11.2,
> option A) are implemented in `converter/hardware_colors.py`; the RGB → hardware colour mapping uses
> the measured palette and the CIE94 match of the `rgb2a8` tool (`converter/atari_match.py`, see
> `src/tests/test_atari_match.py`). The measurements and reasoning below are unchanged and remain the
> reference for why the fix looks the way it does. `DRKMSK` masking (§11.5) is still not compensated.
>
> Date: 2026-11-06. Related: `docs/GUI_IMPLEMENTATION_PLAN.md` §3.3 (B5), T0.3, T6.1.
> Every claim below is tagged **(a) measured**, **(b) hardware documentation** or **(c) decision**;
> see §1.1.

## 1. Question

`converter/exporters/xex_export.py` builds a bootable XEX viewer. Its init routine writes five values
to `$02C4..$02C8` (`COLOR0..COLOR4`). The plan suspected (B5) that the background belongs in `COLBK`
(`$02C8`) and that the colours therefore appear "shifted by one" on hardware. The same file's ASM
export comment calls the table "COLBK, COLPF0-PF3", which contradicts the XEX order.

T0.3 had to answer, with emulator evidence:

1. Which GTIA registers does the XEX init actually write (directly or through OS shadows)?
2. Which register does each ANTIC pixel value use in modes D/E/F?
3. What does the user therefore see, per mode?

### 1.1 Evidence classification

| Tag | Meaning |
|---|---|
| **(a) measured** | produced in the `altirra-bridge` session and stored in `docs/evidence/xex-color-registers/` |
| **(b) documented** | taken from Atari hardware documentation; *not* measured in this environment |
| **(c) decision** | requires a design decision; intentionally left open by this task |

| # | Statement | Category | Where |
|---|---|---|---|
| 1 | The XEX init writes only `SDMCTL`, `SDLSTL/H` and `COLOR0-4`, then loops forever | (a) | §4, §5 |
| 2 | The OS VBI copies `COLOR0-4 → COLPF0-3 + COLBK` and `PCOLR0-3 → COLPM0-3`, every frame | (a) | §5 |
| 3 | `DRKMSK = $FE` clears bit 0 of every copied colour (`$AB` is displayed as `$AA`) | (a) | §9 |
| 4 | In modes D/E, pixel values 0-3 select `COLBK`, `COLPF0`, `COLPF1`, `COLPF2` | (b), corroborated by (a) | §7.1 |
| 5 | In mode F, `COLPF2` supplies the hue and the background luminance, `COLPF1` the foreground luminance | (b) | §7.2 |
| 6 | Mode F therefore cannot display two arbitrary, independent colours | (b), consequence of 5 | §7.2, §8.2 |
| 7 | The current XEX renders D/E images shifted by one playfield register (3 visible colours instead of 4) | (a) | §8.1 |
| 8 | The current XEX renders an invisible (all black) ANTIC F picture | (a) | §8.2 |
| 9 | Proposed D/E assignment (`palette[0]→COLBK`, `palette[1..3]→COLPF0..COLPF2`) | (c) proposal | §11.1 |
| 10 | How ANTIC F should map two palette colours | (c) open | §11.2 |
| 11 | Whether XEX export should pre-adjust odd indices for `DRKMSK` | (c) open | §11.5 |

## 2. Tooling (mandatory skill)

| Item | Value |
|---|---|
| Skill | `.agents/skills/altirra-bridge/SKILL.md` (+ `references/python-quickstart.md`) |
| Emulator | `…\AltirraSDL\build\windows-sdl-release\src\AltirraSDL\Release\AltirraSDL.exe` (build 2026-10-28, 16 095 744 B) |
| Bridge SDK | `altirra_bridge` 0.1.0, editable install from `…\AltirraSDL\src\AltirraSDL\AltirraBridge\sdk\python` |
| Machine | `machine=800XL`, `memory=320K`, `video=ntsc`, `basic=false`, XL OS kernel `$942fa929926f84c3` (`evidence/…/emulator-config.json`) |

Bridge operations used (all through `altirra_bridge.AltirraBridge`):

| Operation | Used for |
|---|---|
| `config()` | record the machine profile above |
| `boot(path)` / command-line boot | load the XEX under test (see §10 limitation) |
| `frame(n)` | deterministic settle before every read |
| `regs()` | confirm the CPU is parked in the XEX init loop (`$3800-$382A`) |
| `gtia()` | decoded GTIA colour registers |
| `antic()` | `DMACTL` / `DLIST` to confirm DMA and the display list |
| `dlist()` | walked display list (mode 14 = ANTIC E/F, `LMS` present) |
| `peek()` / `peek16()` | `COLOR0-4` shadows, `PCOLR0`, `COLRSH`, `DRKMSK`, `RUNAD` |
| `memdump()` | init bytes, OS VBI bytes, VRAM samples |
| `disasm()` | init routine and the OS VBI copy loop |
| `watch_set()` / `bp_list()` / `bp_clear_all()` / `resume()` | prove the OS rewrites the colour registers every frame |
| `hwpoke()` | prove `gtia()` reads the hardware latch (not the shadow) |
| `poke()` | shadow-effect experiment |
| `screenshot()` | user-visible result per mode |
| `palette()` | emulator RGB table, used to identify screenshot colours |

## 3. Artefacts under test

Generated with the unmodified CLI (no source change):

```powershell
python convert.py src/tests/fixtures/reference.png <out>/e/screen.xex --export xex --dither none --palette median_cut
python convert.py src/tests/fixtures/reference.png <out>/d/screen.xex --export xex --dither none --palette median_cut --width 160 --height 96
python convert.py src/tests/fixtures/reference.png <out>/f/screen.xex --export xex --dither none --palette median_cut --width 320 --height 192 --colors 2
```

| Case | Image | Palette (`median_cut`) | XEX segments | `RUNAD` |
|---|---|---|---|---|
| ANTIC E | 160×192, 4 colours | `[0, 236, 178, 171]` = `$00 $EC $B2 $AB` | `$4000-$5E0F`, `$3E00-$3EC9`, `$3800-$382A` | `$3800` |
| ANTIC D | 160×96, 4 colours | same | `$4000-$4EFF`, `$3E00-$3E67`, `$3800-$382A` | `$3800` |
| ANTIC F | 320×192, 2 colours | `[0, 1]` = `$00 $01` | `$4000-$5E0F`, `$3E00-$3EC9`, `$3800-$382A` | `$3800` |

## 4. What the XEX init actually writes

Init routine, 43 bytes at `$3800` (measured with `memdump(0x3800, 0x2B)`, decoded with `disasm`):

```
a9228d2902 a9008d3002 a93e8d3102 a9008dc402 a9ec8dc502
a9b28dc602 a9ab8dc702 a9008dc802 4c0038
```

| Instruction | Target | Value | Meaning |
|---|---|---|---|
| `LDA #$22 / STA $022F` | `SDMCTL` shadow | `$22` | enable DL DMA, narrow playfield |
| `LDA #$00 / STA $0230` | `SDLSTL` | `$00` | display list = `$3E00` |
| `LDA #$3E / STA $0231` | `SDLSTH` | `$3E` | |
| `LDA #$00 / STA $02C4` | `COLOR0` | `$00` | palette[0] |
| `LDA #$EC / STA $02C5` | `COLOR1` | `$EC` | palette[1] |
| `LDA #$B2 / STA $02C6` | `COLOR2` | `$B2` | palette[2] |
| `LDA #$AB / STA $02C7` | `COLOR3` | `$AB` | palette[3] |
| `LDA #$00 / STA $02C8` | `COLOR4` | `$00` | **5th value = padding** (`$00`) |
| `JMP $3800` | – | – | re-runs the whole init forever |

Confirmed at run time: `PC` sits between `$380F` and `$3824` in every sample, and the shadows read
back exactly as written (`antic-e-state.json`: `shadows_02c0_02c8 = 00 00 00 00 | 00 EC B2 AB 00`).
Nothing is written to `$D012-$D01A` (GTIA) or to `$02C0-$02C3` (`PCOLR0-3`) by the program itself.

## 5. What the OS does with those shadows (measured, decisive)

The OS VBI copies nine bytes into the GTIA colour registers every frame. Disassembly at `$E80E`
(measured `memdump` bytes `58 bd c0 02 45 4f 25 4e 9d 12 d0 ca 10 f2`):

```asm
$E80E  cli
$E80F  lda $02C0,X        ; X = 8,7,…,0  ->  reads $02C8 … $02C0
$E812  eor COLRSH  ($4F)  ; measured: $00
$E814  and DRKMSK  ($4E)  ; measured: $FE  (clears bit 0 of every colour!)
$E816  sta $D012,X        ; -> writes COLBK … COLPM0
$E819  dex
$E81A  bpl $E80E
```

Evidence that this loop runs continuously and that `gtia()` reports hardware state:

- a write watchpoint on `$D01A` (`COLBK`) hits **inside this loop** (`PC=$E819`) on every frame;
- `hwpoke(0xD017, $1E)` is visible in `gtia()` **immediately** (`COLPF1: $EC → $1E`), so `gtia()`
  reads the latch, not the shadow;
- a `poke` of `$02C5` is reverted within one frame — because the XEX init ends with `JMP $3800` and
  rewrites the shadows continuously (this also explains why `poke` experiments look "ineffective").

Resulting **shadow → register mapping** (verified, not assumed):

| Shadow | Address | GTIA register |
|---|---|---|
| `PCOLR0-PCOLR3` | `$02C0-$02C3` | `COLPM0-COLPM3` (`$D012-$D015`) |
| `COLOR0` | `$02C4` | **`COLPF0`** (`$D016`) |
| `COLOR1` | `$02C5` | **`COLPF1`** (`$D017`) |
| `COLOR2` | `$02C6` | **`COLPF2`** (`$D018`) |
| `COLOR3` | `$02C7` | **`COLPF3`** (`$D019`) |
| `COLOR4` | `$02C8` | **`COLBK`** (`$D01A`) |

So the XEX fills `COLPF0, COLPF1, COLPF2, COLPF3, COLBK` with `palette[0..3]` + a hard-coded `$00`
padding — not with "background + playfield colours".

## 6. Measured register state per mode

`gtia()` right after `frame(60)` following boot (`antic-*-state.json`, verbatim):

| Case | `COLOR0-4` shadows | COLPF0 | COLPF1 | COLPF2 | COLPF3 | COLBK |
|---|---|---|---|---|---|---|
| ANTIC E | `00 EC B2 AB 00` | `$00` | `$EC` | `$B2` | `$AA` | `$00` |
| ANTIC D | `00 EC B2 AB 00` | `$00` | `$EC` | `$B2` | `$AA` | `$00` |
| ANTIC F | `00 01 00 00 00` | `$00` | `$00` | `$00` | `$00` | `$00` |

(`COLPM0-3 = $00` in all cases; `DMACTL=$22`, `DLIST=$3E00`, `NMIEN=$40`.)

## 7. Register usage per ANTIC mode

The two mode families must be kept apart: D/E are four-colour playfields with independent registers,
F is a single-hue, two-luminance playfield.

### 7.1 ANTIC D and E (2bpp, four independent colours) — **(b)** documented, **(a)** corroborated

| Pixel value | Register |
|---|---|
| 0 | `COLBK` (`$D01A`) |
| 1 | `COLPF0` (`$D016`) |
| 2 | `COLPF1` (`$D017`) |
| 3 | `COLPF2` (`$D018`) |

`COLPF3` is not used by modes D/E. This is the standard 4-colour GTIA mapping; the ANTIC E screenshot
corroborates it, because it shows exactly the three logical colours that the measured registers
predict (§8.1).

### 7.2 ANTIC F (1bpp) — **(b)** documented: one hue, two luminances

Mode F does **not** provide two independent colour registers. Its two pixel values are built from two
registers that share the hue:

| Pixel value | Hue (chrominance) | Luminance |
|---|---|---|
| 0 (background) | `COLPF2` | `COLPF2` |
| 1 (foreground) | `COLPF2` | `COLPF1` |

Consequences that matter for the converter:

- mode F can only render **two luminances of a single hue**; it cannot represent two arbitrary,
  independent RGB colours;
- `COLPF2` therefore carries two roles at once (hue source and background luminance), while `COLPF1`
  contributes luminance only;
- `COLPF0`, `COLPF3` and `COLBK` are not the playfield colours in mode F (`COLBK` stays the border /
  area colour);
- the natural D/E assignment `palette[0] → COLOR4 (COLBK)`, `palette[1] → COLOR0 (COLPF0)` (§11.1) is
  **not** a valid fix for F: both registers are outside the mode-F playfield path, so such a "fix"
  would still produce an invisible picture.

Status of this subsection: **(b)** documentation only — the emulator run cannot corroborate it here,
because the exported F values (`$00`, `$01`) collapse to `$00` in every register involved (§8.2). It
must be re-verified in T6.1 with a register set that makes the hue/luminance split visible.

## 8. What the user sees (verdict on B5)

### 8.1 ANTIC E and D — colours shifted by one playfield register **(a)**

Intended versus measured (palette `$00 $EC $B2 $AB`):

| Pixel value | Intended colour | Register actually used | Value there | Result |
|---|---|---|---|---|
| 0 | `$00` (palette[0]) | `COLBK` | `$00` | ✅ correct |
| 1 | `$EC` (palette[1]) | `COLPF0` | `$00` | ❌ black instead of `$EC` |
| 2 | `$B2` (palette[2]) | `COLPF1` | `$EC` | ❌ shows palette[1] |
| 3 | `$AB` (palette[3]) | `COLPF2` | `$B2` | ❌ shows palette[2] |
| – | `$AB` never displayed | `COLPF3` | `$AA` | ❌ modes E/D do not use `COLPF3` |

The screenshot confirms it: `antic-e-screen.png` and `antic-d-screen.png` contain only **three**
logical colours — `$00` (`(0,0,0)`), `$B2` (`(0,53,18)`, exact match in the emulator palette table)
and `$EC` (`(191,216,87)`; nearest `$EC=(190,217,86)`, distance 1.7). The other RGB values in the
histogram are vertical blend artefacts of Altirra's scaled 336×448 screenshot (`(0,40,14)` is ≈75 %
of `$B2`, `(144,162,66)` ≈75 % of `$EC`). **`$AB` (`(69,202,241)`, light blue) never appears.**

### 8.2 ANTIC F — no visible picture; two independent colours are impossible **(a)** + **(b)**

The converter selects 2 colours (`$00`, `$01`, `median_cut`) and writes `COLOR0=$00`, `COLOR1=$01`,
`COLOR2..COLOR4=$00`. After the OS copy (§5, §9) every colour register reads `$00`
(`antic-f-state.json`: `COLPF0=$00, COLPF1=$00, COLPF2=$00, COLPF3=$00, COLBK=$00`).

Applying the mode-F model from §7.2: background = `COLPF2` = `$00`; foreground = `COLPF1` luminance
(`$00`) with `COLPF2` hue (`$00`). Both pixel values render as `$00`, and `antic-f-screen.png` confirms
it: **150 528 pixels, 1 distinct RGB value — a completely black screen.**

Two different problems coincide in the F case:

1. **(a) measured / register order** — the exported values never reach `COLPF1`/`COLPF2` in a usable
   form: the values land in `COLPF0`/`COLPF1` instead of the mode-F pair, and the one non-zero value
   (`$01`) is additionally masked to `$00` by `DRKMSK` (§9).
2. **(b) documented / design-level** — even with a corrected register order, mode F cannot show two
   arbitrary palette colours. The converter's 2-colour palette may pick any two Atari colours
   (`$00` and `$01` here), while mode F can only show two luminances of one hue. Choosing what to do
   is a design decision, not a bug fix (§11.2).

## 9. Additional finding **(a)**: `DRKMSK = $FE` clears bit 0 of every colour

The VBI applies `and DRKMSK` with the measured value `$FE` before writing each colour register, so
every colour index loses its luminance LSB on the way to GTIA:

- stored `$AB` (171) is displayed as `$AA` (170) — exactly the 1-unit difference measured between
  `COLOR3` and `COLPF3`;
- any odd index (e.g. `$65`, `$01`, `--atari-colors 0x03,…`) is displayed one luminance step lower.

`COLRSH` (measured `$00`) is the companion shift value; it is a no-op here but would also alter the
displayed colours if an application ever sets it.

## 10. Tooling limitations and uncertainties

- **`BOOT` wedge.** Issuing the bridge's `BOOT <xex>` command sometimes wedged the emulator: the SDL
  main loop stopped right after `BOOT dispatched`, the window turned "not responding" and the bridge
  stopped answering. It was reproduced with an unrelated third-party XEX
  (`atari-call-of-the-wintermoon/call_of_the_wintermon_en.xex`), so **it is not caused by our file**.
  The cause inside the emulator build was not determined. All results in this document were therefore
  taken with an alternative, equally legitimate boot path: start
  `AltirraSDL.exe --bridge <screen.xex>` (file loaded at start-up) and attach through the bridge.
- **Screenshot geometry.** `screenshot()` returns 336×448 with vertical blending, so one logical Atari
  colour can appear as two RGB values. Colour identification was done against the emulator's own
  `palette()` table and is reported as "nearest index", not as exact semantics.
- **Mode-F model is documentation-based (§7.2).** The emulator run could only confirm the *end result*
  (all black), because the exported values for this case collapse to `$00` in every register involved.
  A conclusive F experiment needs a register pair with different hue/luminance (`COLPF2=$3A`,
  `COLPF1=$0E`) and is part of T6.1.
- **Single configuration.** Measured on `machine=800XL, video=ntsc` with XL OS kernel
  `$942fa929926f84c3`. The `lda $02C0,X / sta $D012,X` VBI copy with `COLRSH`/`DRKMSK` is standard
  XL/XE behaviour, but **other OS revisions (e.g. 400/800 OS-B) and PAL machines were not tested**,
  and `DRKMSK`/`COLRSH` values may differ there. Nothing here was verified on real hardware.
- **Not investigated:** the exact rendering path that produces the blend colours in the screenshots
  (not relevant to B5). The `A` register values sampled inside the init loop are simply the operands
  of the `LDA #imm` instructions being executed.

## 11. Recommendation for T6.1 (not implemented — items 9-11 of §1.1 are decisions)

### 11.1 ANTIC D/E — proposed register assignment **(c)** proposal

In `_build_init_routine`: `palette[0] → $02C8 (COLBK)`, `palette[1] → $02C4 (COLPF0)`,
`palette[2] → $02C5 (COLPF1)`, `palette[3] → $02C6 (COLPF2)`. Do **not** touch `$02C0-$02C3`
(player/missile shadows) — they are unrelated to the playfield.

### 11.2 ANTIC F — must be handled separately, and it needs a design decision **(c)** open

`palette[0] → COLOR4`, `palette[1] → COLOR0` (the D/E fix) is **not** a valid F solution: mode F takes
its playfield from `COLPF2` (hue + background luminance) and `COLPF1` (foreground luminance) only
(§7.2). Because F cannot display two arbitrary colours, one of the following has to be chosen — and
documented — before any F code is written:

| Option | Idea | Trade-off |
|---|---|---|
| A | Derive a single hue for `COLPF2` and two luminances for `COLPF1`/`COLPF2` from the two palette colours | Always visible, but the image is lossy and "wrong" by design |
| B | Validate/warn: require the two palette colours to share a hue, otherwise tell the user F is unsuitable | Honest, keeps fidelity, but pushes work to the user |
| C | Do not offer the XEX viewer for F (keep the data exports only) | Simplest, but drops a feature |
| D | Bypass the OS shadows and write GTIA from a DLI/VBI per line | Enables more colours later, much larger change, out of scope for T6.1 |

Whatever is chosen, T6.1 must re-verify the F model empirically with a register pair that makes the
hue/luminance split visible (e.g. `COLPF2=$3A`, `COLPF1=$0E`), because §7.2 is documentation-based.

### 11.3 Exported documentation to correct

The ASM export comment (`"; Color register values (COLBK, COLPF0-PF3)"`) must describe the order the
XEX actually uses, and the `--atari-colors` order has to be documented explicitly (suggestion:
`COLBK, COLPF0, COLPF1, COLPF2[, COLPF3]`).

### 11.4 Tests and golden impact

- the `.xex` bytes change (golden cases `all_formats_atkinson`, `xex_extra_mode_d`,
  `xex_extra_mode_e_190`, `xex_extra_mode_f`); `screen.bin/.png/.s/.dta` stay untouched, so the T6.1
  commit must show a golden diff limited to `.xex` entries;
- add a regression test that inspects the emitted init bytes (order of the `STA` targets) — no
  emulator required;
- for ANTIC F, add a test that pins whichever strategy §11.2 lands on.

### 11.5 Open question: `DRKMSK` masking **(c)** open

Even after the fix, odd colour indices lose bit 0 (§9). Options: document it; pre-round exported XEX
colour indices to even luminance; or bypass the shadows entirely (option D above).

## 12. Reproduction

```powershell
# 1. build the artefact (unmodified CLI)
python convert.py src/tests/fixtures/reference.png $env:TEMP\t03\e\screen.xex --export xex --dither none --palette median_cut

# 2. start the emulator with the file on the command line (avoids the BOOT wedge, §10)
$exe = 'C:\Users\grzes\Documents\Projects\AltirraSDL\build\windows-sdl-release\src\AltirraSDL\Release\AltirraSDL.exe'
$p = Start-Process -FilePath $exe -ArgumentList '--bridge', "$env:TEMP\t03\e\screen.xex" -PassThru
Start-Sleep -Seconds 10

# 3. inspect through the skill's Python SDK
$env:PYTHONPATH = 'C:\Users\grzes\Documents\Projects\AltirraSDL\src\AltirraSDL\AltirraBridge\sdk\python'
python -c "from altirra_bridge import AltirraBridge; a = AltirraBridge.from_token_file(r'C:\Users\grzes\AppData\Local\Temp\altirra-bridge-$($p.Id).token'); a.frame(60); print(a.peek(0x02C0, 9).hex()); g = a.gtia(); print(g['COLPF0'], g['COLPF1'], g['COLPF2'], g['COLBK']); open('shot.png', 'wb').write(a.screenshot()); a.quit()"
```

Expected (current, defective) output: shadows `0000000000ecb2ab00`, registers `$00 $EC $B2 $00` for
`COLPF0/1/2/COLBK` — the shift documented in §8.1.

## 13. Evidence files

All under `docs/evidence/xex-color-registers/`:

| File | Content |
|---|---|
| `antic-e-screen.png`, `antic-d-screen.png`, `antic-f-screen.png` | emulator screenshots (336×448) |
| `antic-e-state.json`, `antic-d-state.json`, `antic-f-state.json` | `regs`, shadows, `gtia()` colours, `antic()`, `dlist` size, init bytes |
| `vbi-probe.json` | OS VBI bytes + disassembly, `COLRSH`/`DRKMSK`, immediate `hwpoke` read-back, watchpoint hit |
| `register-probes.json` | shadow-poke effects, direct `hwpoke` effects, relevant emulator palette RGB values |
| `emulator-config.json` | machine profile (`800XL`, `320K`, `ntsc`, kernel ids) |

