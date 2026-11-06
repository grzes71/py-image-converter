# Bucket Depth

## 18.3 Bucket Sort — 3D Depth Order

An 8-way bucket sort is the usual pick for the ~50–100 visible faces of a
filled-3D object. Depth keys are 0–255 unsigned bytes, so the bucket index
is `depth >> 5` (256 / 8 = 32 depth units per bucket). Zero comparisons.

> The shift must match the bucket count: `>> 5` gives 8 buckets, `>> 3`
> would give 32. Using `>> 3` with only 8 bucket heads (below) silently
> drops the faces that land in buckets 8..31 — a classic mistake.

```asm
SortVisible
        ldx    #$00
        lda    #$00
        sta    bkt_0..bkt_7

@fill   lda    BUF_DEPTH,x
        lsr                   ; depth >> 5  -> bucket 0..7
        lsr
        lsr
        lsr
        lsr
        tay
        lda    bkt_head,y     ; current tail
        sta    next,x
        lda    vis_quad,x
        sta    bkt_head,y     ; push into bucket
        inx
        cpx    vis_count
        bne    @fill

        ; Drain bucket 7 (far) to bucket 0 (near)
        ldy    #$07
@bucket lda    bkt_head,y
        beq    @next
        ; follow linked list into sorted output array
@next   dey
        bpl    @bucket
```

| | |
|---|---|
| Buckets | 8 (depth >> 5) |
| Bucket storage | BKT_HEAD[8] ZP |
| Link storage | NXT_PTR[96] |
| Cycles | ~500 (96 verts, single pass) |
| Comparisons | 0 |

---
