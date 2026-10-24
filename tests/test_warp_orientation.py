import cv2
import numpy as np
import pytest

from ReferenceSurfaceMeasurer import ReferenceSurfaceMeasurer


@pytest.mark.parametrize("reference_size", [(200, 300), (300, 200), (250, 250)])
@pytest.mark.parametrize("landscape", [False, True])
def test_warp_preserves_photo_orientation_and_measurement_scale(reference_size, landscape):
    width, height = (600, 400) if landscape else (400, 600)
    # Distinct corners reveal quarter turns, half turns, and reflections.
    colors = [(30, 60, 220), (40, 210, 70), (220, 80, 40), (60, 220, 220)]
    flat = np.zeros((height, width, 3), dtype=np.uint8)
    flat[:height // 2, :width // 2] = colors[0]
    flat[:height // 2, width // 2:] = colors[1]
    flat[height // 2:, width // 2:] = colors[2]
    flat[height // 2:, :width // 2] = colors[3]
    corners = np.float32([[0, 0], [width - 1, 0],
                          [width - 1, height - 1], [0, height - 1]])
    quad = np.float32([[60, 30], [width + 25, 55],
                       [width + 50, height + 60], [35, height + 25]])
    photo = cv2.warpPerspective(flat, cv2.getPerspectiveTransform(corners, quad),
                                (width + 100, height + 100))
    measurer = ReferenceSurfaceMeasurer(reference_size_mm=reference_size,
                                        save_debug_images=False)
    # Detection does not guarantee any particular starting corner or winding.
    corrected, transform = measurer._warp_reference(photo, quad[[2, 0, 3, 1]])
    h, w = corrected.shape[:2]
    for (x, y), color in zip([(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)], colors):
        np.testing.assert_allclose(corrected[int(y * h), int(x * w)], color, atol=2)
    mapped = cv2.perspectiveTransform(quad[None], transform)[0]
    np.testing.assert_allclose(mapped, [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], atol=1e-3)

    physical_w, physical_h = sorted(reference_size, reverse=landscape)
    assert w / h == pytest.approx(physical_w / physical_h, rel=0.01)
    # Check both the garment bounding-box and small-object rotated-box paths.
    for fraction in (0.2, 0.6):
        bw, bh = round(w * fraction), round(h * fraction)
        contour = np.int32([[[10, 10]], [[10 + bw - 1, 10]],
                            [[10 + bw - 1, 10 + bh - 1]], [[10, 10 + bh - 1]]])
        measurement = measurer._measurement_from_contour(contour, (w, h))
        expected = sorted([fraction * physical_w, fraction * physical_h])
        assert sorted([measurement.width_mm, measurement.height_mm]) == pytest.approx(expected, abs=2)


def test_warp_recomputes_axes_when_measurer_is_reused():
    measurer = ReferenceSurfaceMeasurer(reference_size_mm=(200, 300), save_debug_images=False)
    image = np.zeros((700, 700, 3), dtype=np.uint8)
    for width, height, expected in [(600, 400, (300, 200)), (400, 600, (200, 300))]:
        quad = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
        measurer._warp_reference(image, quad)
        assert measurer._warp_reference_size_mm == expected
    assert measurer.reference_size_mm == (200, 300)
