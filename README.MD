
```bash
pip install --upgrade --force-reinstall --no-cache-dir .
```

## Photo and contour comparison

Each case shows the original photo, the perspective-corrected photo, and the
extracted contour side by side.
The contour panel includes the measurement box, with blue width and orange
height arrows and values in millimeters.
Each card shows the latest test outcome, expected measurements, tolerance, and
measurement errors. The header records the run time and selected suite totals.
A dashed magenta box shows the expected size, centered on the measured box and
drawn at the same scale when expected dimensions are available.

Perspective correction preserves the input photo's orientation. The reference
dimensions are matched to the visible horizontal and vertical edges, so either
portrait or landscape photos can use the same reference size. Measurement axes
follow the corrected photo. Inputs are decoded image arrays; any camera rotation
metadata must be applied when loading the photo.

Foreground segmentation estimates background color and its variation from the
reference border. Lab channels with a wide background range receive less weight,
so uneven illumination on the canvas does not add bright or shaded patches to a
pale shirt's contour. The debug trace records the background range and channel
weights alongside the segmentation threshold.

![Photo and contour comparison](assets/contour-comparison.jpg)

Regenerate this asset with the full test suite:

```bash
OBJECT_MEASURER_SAVE_DEBUG_IMAGES=false python -m pytest
```

The image is saved to `assets/contour-comparison.jpg`. Running a subset of the
photo tests replaces it with only the selected cases; run the full suite before
checking in the asset.
