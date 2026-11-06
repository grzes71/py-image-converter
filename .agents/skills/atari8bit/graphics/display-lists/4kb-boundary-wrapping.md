# 4KB Memory Boundary Wrapping & LMS Segment Alignment

## §7  ANTIC 12-Bit Counter Limit & 4KB Memory Boundary Wrapping

ANTIC's internal video memory address counter is **12-bit** (`$0000`–`$0FFF` / 4096 bytes / 4 KB). When ANTIC fetches screen bytes sequentially across scanlines, only the bottom 12 bits increment automatically. The high 4 bits (bits 12–15, loaded by a Load Memory Scan / LMS instruction) **do not increment** when bits 0–11 wrap from `$FFF` to `$000`.

### Root Cause Mechanics
1. **12-Bit Counter Wrap:** If ANTIC reaches `$xFFF` during scanline rendering without an LMS instruction, the address counter wraps to `$x000` (the start of the same 4 KB memory page) instead of advancing to `$x+1000`.
2. **Mode Scanline Math:**
   - **Mode E / Graphics 15 (160×192 / 160×190, 2 bpp):** 40 bytes/scanline. `floor(4096 / 40) = 102 lines` (`102 * 40 = 4080 bytes`). Line 102 (the 103rd scanline) crosses the 4 KB boundary `$x000`.
   - **Mode F / Graphics 8 (320×192 / 320×190, 1 bpp):** 40 bytes/scanline. `floor(4096 / 40) = 102 lines` (`102 * 40 = 4080 bytes`).
   - **Mode D / Graphics 7 (160×96, 2 bpp):** 40 bytes/scanline. `96 * 40 = 3840 bytes` (< 4096 bytes), fits in a single 4 KB segment without wrapping.

---

## The Mid-Scanline Wrapping Gotcha (Horizontal Shift)

A common mistake is placing an LMS instruction at scanline 102 reloading address `IMAGE_BUFFER + 4080` (e.g. `$4FF0` if `IMAGE_BUFFER = $4000`):

- **Problem:** Address `$4FF0` has only 16 bytes remaining before the `$5000` 4 KB boundary (`$4FF0`–`$4FFF`).
- **Mid-Line Wrap:** Scanline 102 requires 40 bytes. ANTIC reads the first 16 bytes from `$4FF0`..`$4FFF`. At byte 16, the 12-bit address counter wraps from `$FFF` to `$000`. High bits remain `$4`, so ANTIC fetches the remaining 24 bytes of scanline 102 from `$4000`..`$4017`!
- **Visual Artifact:** The display splits horizontally—the first 64 pixels (16 bytes) come from the right edge of line 102, and the rest of the scanline (and subsequent scanlines) wraps to the left margin shifted by 16 bytes.

---

## 4KB Alignment & Memory Padding Solution

To ensure no scanline straddles a `$x000` boundary:
1. **Pad VRAM Segments in Memory:** Pad each 4 KB VRAM block in RAM to exactly 4096 bytes.
   - Segment 1: 102 scanlines × 40 bytes = 4080 bytes (`$4000`–`$4FEF`).
   - Padding: Append 16 zero-padding bytes (`$4FF0`–`$4FFF`).
   - Segment 2: Starts cleanly at offset 4096 (`$5000`).
2. **Reload LMS at 4KB Aligned Address:** Set the Display List LMS instruction for Segment 2 to point to `$5000` (`a(IMAGE_BUFFER + 4096)`).

### MADS Assembly Example

```mads
DLIST_PREVIEW
    ; --- Upper Margin ---
    dta $70, $70, $70               ; 24 blank lines

    ; --- Segment 1: Lines 0..101 (102 lines = 4080 bytes) ---
    dta $4E, a(IMAGE_BUFFER + 0)    ; LMS Mode E: start of VRAM ($4000)
    .rept 101
    dta $0E                         ; Mode E scanlines
    .endr

    ; --- Segment 2: Lines 102..189 (88 lines = 3520 bytes) ---
    dta $4E, a(IMAGE_BUFFER + 4096) ; LMS Mode E: reload aligned at 4KB offset ($5000)
    .rept 87
    dta $0E                         ; Mode E scanlines
    .endr

    ; --- Jump and Wait for Vertical Blank ---
    dta $41, a(DLIST_PREVIEW)
```

---

## Dynamic 4KB Segment Generator (Python Implementation)

```python
def build_dlist(
    mode: str,
    height: int = 192,
    width: int = 160,
    screen_base_address: int = 0x4000,
    pad_4kb: bool = True,
) -> bytearray:
    mode_byte = 0x0E if mode == "E" else (0x0F if mode == "F" else 0x0D)
    lms_byte = mode_byte | 0x40

    bits_per_pixel = 1 if mode == "F" else 2
    bytes_per_line = (width * bits_per_pixel) // 8
    max_lines_per_segment = 4096 // bytes_per_line  # 102 lines for 40 B/line

    dlist = bytearray([0x70, 0x70, 0x70])  # Blank lines margin

    remaining_lines = height
    current_byte_offset = 0

    while remaining_lines > 0:
        lines_in_segment = min(remaining_lines, max_lines_per_segment)
        segment_bytes = lines_in_segment * bytes_per_line

        # Assertion: Verify segment size does not exceed 4KB limit
        assert segment_bytes <= 4096, f"Segment size {segment_bytes} bytes exceeds 4KB limit"

        target_addr = screen_base_address + current_byte_offset
        dlist.extend([lms_byte, target_addr & 0xFF, (target_addr >> 8) & 0xFF])

        if lines_in_segment > 1:
            dlist.extend([mode_byte] * (lines_in_segment - 1))

        if pad_4kb and remaining_lines > lines_in_segment:
            current_byte_offset += 4096  # Advance to next 4KB boundary ($x000)
        else:
            current_byte_offset += segment_bytes

        remaining_lines -= lines_in_segment

    dlist.extend([0x41, screen_base_address & 0xFF, (screen_base_address >> 8) & 0xFF])
    return dlist
```

---

## Quick Reference Summary

| Parameter | Mode D (7) | Mode E (15) | Mode F (8) |
|---|---|---|---|
| Resolution | 160×96 | 160×192 / 160×190 | 320×192 / 320×190 |
| Color Depth | 2 bpp (4 col) | 2 bpp (4 col) | 1 bpp (2 col) |
| Bytes / Line | 40 | 40 | 40 |
| Max Lines / 4KB | 102 | 102 | 102 |
| Total Segments | 1 | 2 | 2 |
| Segment 1 Lines | 96 | 102 | 102 |
| Segment 1 LMS | `$4D, $00, $40` | `$4E, $00, $40` | `$4F, $00, $40` |
| Segment 2 Lines | N/A | 90 (or 88) | 90 (or 88) |
| Segment 2 LMS | N/A | `$4E, $00, $50` | `$4F, $00, $50` |
