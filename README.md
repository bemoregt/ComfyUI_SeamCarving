# ComfyUI Seam Carving Resize

A ComfyUI custom node for **content-aware image resizing** using the [Seam Carving algorithm](https://dl.acm.org/doi/10.1145/1275808.1276390) (Avidan & Shamir, 2007).

Unlike standard resizing, Seam Carving removes or inserts the *least visually important* paths through the image, preserving subjects and structure while eliminating low-energy regions like sky, background, and whitespace.

![이미지 스펙트럼 예시](https://github.com/bemoregt/ComfyUI_SeamCarving/blob/main/ScrShot%201.png)

![이미지 스펙트럼 예시](https://github.com/bemoregt/ComfyUI_SeamCarving/blob/main/ScrShot%202.png)

---

## How It Works

A **seam** is a connected, one-pixel-wide path from the top to the bottom (vertical) or left to right (horizontal) of an image, where each step moves to an adjacent pixel.

1. **Energy map** — Each pixel is assigned an energy score using a gradient magnitude (central difference). High-energy pixels correspond to edges and important content; low-energy pixels correspond to flat, unimportant regions.

2. **Seam finding** — A dynamic programming pass accumulates the minimum-cost path from one edge of the image to the other.

3. **Shrinking** — The lowest-energy seam is found and removed, one at a time, until the target dimension is reached.

4. **Enlarging** — `k` optimal seams are located on the original image first (using a column-map to track original coordinates), then inserted all at once as averaged pixels. This prevents the banding artifacts caused by naively reinserting the same seam repeatedly.

5. **Height adjustment** — The image is transposed so that horizontal seam operations reuse the same vertical seam logic.

---

## Node Parameters

| Parameter | Type | Range | Default | Description |
|---|---|---|---|---|
| `image` | IMAGE | — | — | Input image tensor `(B, H, W, C)` |
| `target_width` | INT | 1 – 8192 | 512 | Output width in pixels |
| `target_height` | INT | 1 – 8192 | 512 | Output height in pixels |

**Output:** `IMAGE` tensor of shape `(B, target_height, target_width, C)`

- Both shrinking and enlarging are supported independently per axis.
- Setting a target equal to the source dimension skips processing for that axis.
- RGB, RGBA, and single-channel images are all supported.
- Each image in a batch is processed independently to the same target size.

---

## Installation

**Option A — Copy into ComfyUI directly:**

```bash
cp -r ComfyUI_seamcarving /path/to/ComfyUI/custom_nodes/
```

**Option B — Symlink (keeps source editable):**

```bash
ln -s /absolute/path/to/ComfyUI_seamcarving \
      /path/to/ComfyUI/custom_nodes/ComfyUI_seamcarving
```

Restart ComfyUI. The node appears in the node menu under **`image/transform`** as **"Seam Carving Resize"**.

---

## Requirements

| Package | Notes |
|---|---|
| `numpy` | Required — all algorithm code uses NumPy only |
| `torch` | Required — provided by ComfyUI |

No additional dependencies. SciPy is **not** required.

---

## Performance Notes

Processing time scales linearly with the number of seams: `|Δwidth| + |Δheight|`. Each seam requires one full `O(H × W)` dynamic programming pass.

| Change magnitude | Expected feel |
|---|---|
| < 100 px per axis | Fast, suitable for interactive use |
| 100 – 300 px per axis | Noticeable delay; a progress bar is shown in ComfyUI |
| > 300 px per axis | Consider downscaling first with a standard resize node |

For large reductions, chaining a conventional resize node before this one and using Seam Carving only for the final refinement step gives the best quality-to-speed tradeoff.

---

## File Structure

```
ComfyUI_seamcarving/
├── __init__.py        # ComfyUI node class and NODE_CLASS_MAPPINGS registration
└── seam_carving.py    # Pure NumPy algorithm (energy map, DP, removal, insertion)
```

---

## References

- Avidan, S., & Shamir, A. (2007). **Seam carving for content-aware image resizing.** *ACM Transactions on Graphics*, 26(3), Article 10. https://doi.org/10.1145/1275808.1276390
