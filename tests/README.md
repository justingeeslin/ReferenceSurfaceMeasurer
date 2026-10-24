Run the suite from the repository root:

```sh
python -m pytest
```

The photo tests generate a single contact sheet at
`assets/contour-comparison.jpg`. This asset can be checked in and is embedded in
the project README. Each case shows the original photo,
the perspective-corrected photo, and the extracted measurement contours on white.
The last two panels use the same coordinates and scale. Cases are labeled with
their fixture name and measurement scale; repeated cases appear only once.
The contour panel frames the box used for measurement, with blue width and
orange height arrows labeled in millimeters. Garments use an axis-aligned box;
small objects use their rotated minimum-area box, with labels on the measured edges.
When expected dimensions are supplied, a dashed magenta box overlays the dominant
contour at the same physical scale. It shares the measured box's center and edge
directions and uses the same expected-axis pairing as the error report. This shows
size differences; the tests do not specify an expected position. Smoke-only cases
have no expected box. Boxes extending beyond the panel are labeled as clipped.

The sheet includes the photo cases selected for the current run, including failed
photo loads, and is written even when measurement assertions fail. Each card shows
the actual pytest outcome, separating smoke checks from measurement checks. For
measurement tests it also shows expected and actual dimensions in millimeters,
the per-axis tolerance, signed differences (actual minus expected), and the amount
beyond tolerance. Expected axes are aligned using the test's allowance for swapped
width and height, and any swap is explicitly labeled. Only the dominant contour
is checked by the current accuracy assertion; contour count is not asserted.

The header includes the run time in America/Chicago and totals for the entire
selected test suite, including tests without photo cards. Results describe this
run only; running a subset replaces the image. Runs without photo tests do not
generate a new sheet.

To skip the individual debug images while still producing the contact sheet:

```sh
OBJECT_MEASURER_SAVE_DEBUG_IMAGES=false python -m pytest
```
