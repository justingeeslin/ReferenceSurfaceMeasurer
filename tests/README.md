Run the suite from the repository root:

```sh
python -m pytest
```

The photo tests generate a single contact sheet at
`build/test-results/contour-comparison.jpg`. Each case shows the original photo,
the perspective-corrected photo, and the extracted measurement contours on white.
The last two panels use the same coordinates and scale. Cases are labeled with
their fixture name and measurement scale; repeated cases appear only once.

The sheet includes the photo cases selected for the current run and is written
even when measurement assertions fail. Its labels describe extraction results,
not test pass/fail status. Runs without photo tests do not generate a new sheet.

To skip the individual debug images while still producing the contact sheet:

```sh
OBJECT_MEASURER_SAVE_DEBUG_IMAGES=false python -m pytest
```
