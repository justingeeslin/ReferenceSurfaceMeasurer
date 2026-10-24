from pathlib import Path

import cv2
import numpy as np
import pytest

from ReferenceSurfaceMeasurer import ReferenceSurfaceMeasurer


@pytest.mark.parametrize("scale", [0.5, 1.0])
def test_light_blue_shirt_excludes_canvas_below_hem(scale):
    path = Path(__file__).resolve().parents[1] / "test-images/ades-light-blue-1/ades-light-blue-1.jpg"
    img = cv2.imread(str(path))
    assert img is not None
    measurer = ReferenceSurfaceMeasurer(
        scale=scale, reference_size_mm=(762, 762), save_debug_images=False,
    )
    measurements, debug = measurer.measure(img, return_debug=True)
    assert debug["status"] == "ok"
    contour = measurements[0].contour
    h, w = debug["imgWarp"].shape[:2]

    # Spatial checks catch the attached canvas patch even if the bounding box
    # happens to pass a loose dimension tolerance. Keep both sleeves and hem.
    assert np.max(contour[:, 0, 1]) / h < 0.86
    for x, y in [(0.18, 0.30), (0.83, 0.33), (0.50, 0.80)]:
        assert cv2.pointPolygonTest(contour, (x * w, y * h), False) > 0
    assert measurements[0].height_mm == pytest.approx(536.575, abs=15)


@pytest.mark.parametrize("background_level, object_level", [(210, 140), (55, 130)])
def test_background_variation_preserves_achromatic_objects(background_level, object_level):
    # Brightness must still distinguish gray objects after weighting channels
    # for illumination variation on either light or dark reference surfaces.
    levels = np.linspace(background_level - 15, background_level + 15, 400)
    warp = np.repeat(levels.astype(np.uint8)[:, None, None], 400, axis=1)
    warp = np.repeat(warp, 3, axis=2)
    warp[80:320, 90:310] = object_level
    measurer = ReferenceSurfaceMeasurer(save_debug_images=False)
    measurer.debug = {"object_candidate_contours": [], "trace": []}
    measurements, contours, _ = measurer._measure_objects(warp)
    assert len(measurements) == 1
    assert cv2.pointPolygonTest(contours[0], (200, 200), False) > 0
    x, y, w, h = measurements[0].bbox
    assert (x, y, x + w, y + h) == pytest.approx((90, 80, 310, 320), abs=5)
