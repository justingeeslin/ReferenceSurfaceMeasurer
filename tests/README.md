Run the suite from the repository root:

```sh
python -m pytest
```

`test_svg_background.py` includes examples of SVGs with an embedded base64 PNG
and a contour polygon above it. Tests decode the PNG and compare every pixel
against the dominant contour's crop from the corrected photo, using square and
rectangular garment references at two scales. Synthetic examples also verify
canvas scaling, image/shape order, and alignment after contour simplification.

These nine cases also generate `assets/svg-background-comparison.png`: one card
per test with the input, the decoded PNG crop, and the rendered returned SVG.
The cards show the actual pytest outcome, pixel/canvas dimensions, and the
empty-input reset result. Failures and skips still receive cards, with unavailable
outputs labeled. The header includes the selected SVG test totals and run time
in America/Chicago. SVG previews use `rsvg-convert` from librsvg; if it is absent,
the card reports the unavailable preview while retaining the input and PNG crop.

Regenerate just this summary with:

```sh
python -m pytest tests/test_svg_background.py
```

Running a subset replaces the summary with the selected SVG cases. Runs without
SVG background cases leave the image unchanged.

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
