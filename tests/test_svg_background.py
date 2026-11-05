import base64
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import pytest

from ReferenceSurfaceMeasurer import ReferenceSurfaceMeasurer


SVG_NS = "{http://www.w3.org/2000/svg}"
TEST_IMAGES_ROOT = Path(__file__).resolve().parents[1] / "test-images"


def decode_png(payload):
    png = base64.b64decode(payload, validate=True)
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    image = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    assert image is not None
    return image


@pytest.mark.parametrize(
    "canvas_size, expected_size, expected_points",
    [
        (None, (21, 31), "0,0 20,0 20,30 0,30"),
        ((84, 62), (84, 62), "0,0 80,0 80,60 0,60"),
        ((10.5, 93), (10.5, 93), "0,0 10,0 10,90 0,90"),
    ],
)
def test_svg_embeds_cropped_photo_below_scaled_contour(
    canvas_size, expected_size, expected_points, svg_background_comparison,
):
    contour = np.int32([[[10, 20]], [[30, 20]], [[30, 50]], [[10, 50]]])
    # An asymmetric photo makes shifts, rotations, and BGR/RGB mistakes visible.
    y, x = np.indices((80, 100))
    photo = np.dstack((2 * x, 3 * y, x + y)).astype(np.uint8)
    crop = photo[20:51, 10:31]
    encoded, png = cv2.imencode(".png", crop)
    assert encoded
    payload = base64.b64encode(png).decode("ascii")

    svg = ReferenceSurfaceMeasurer()._contour_to_svg(
        contour, canvas_size=canvas_size, image_base64=payload,
    )
    svg_background_comparison(
        title=f"Synthetic photo | canvas {expected_size[0]:g} x {expected_size[1]:g}",
        source=photo, bounds=(10, 20, 21, 31), image_base64=payload, svg=svg,
        notes=["Green box: crop bounds in the input photo.",
               "Checks: PNG pixels, layer order, and image / polygon scaling."],
    )

    root = ET.fromstring(svg)
    image, polygon = list(root)
    assert image.tag == f"{SVG_NS}image"
    assert polygon.tag == f"{SVG_NS}polygon"
    assert image.attrib["id"] == "object-photo"
    assert image.attrib["x"] == image.attrib["y"] == "0"
    assert (float(image.attrib["width"]), float(image.attrib["height"])) == expected_size
    assert tuple(map(float, root.attrib["viewBox"].split())) == (0, 0, *expected_size)
    assert image.attrib["preserveAspectRatio"] == "none"
    prefix, embedded_payload = image.attrib["href"].split(",", 1)
    assert prefix == "data:image/png;base64"
    np.testing.assert_array_equal(decode_png(embedded_payload), crop)
    assert polygon.attrib["id"] == "object-contour"
    assert polygon.attrib["fill"] == "none"
    assert polygon.attrib["points"] == expected_points


def test_simplifying_contour_keeps_original_photo_crop_coordinates(svg_background_comparison):
    mask = np.zeros((300, 300), dtype=np.uint8)
    mask[30:251, 20:221] = 255
    # Simplification removes this one-pixel tip above the rectangle. The photo
    # still starts at y=29, so the simplified top edge must remain at y=1.
    mask[29, 120] = 255
    contour = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0][0]
    assert len(contour) > 250
    encoded, png = cv2.imencode(".png", mask[29:251, 20:221])
    assert encoded

    svg = ReferenceSurfaceMeasurer()._contour_to_svg(
        contour, image_base64=base64.b64encode(png).decode("ascii"),
    )
    svg_background_comparison(
        title="Contour simplification | preserve crop frame",
        source=mask, bounds=(20, 29, 201, 222),
        image_base64=base64.b64encode(png).decode("ascii"), svg=svg,
        notes=["840 contour points simplify to 4; crop includes the one-pixel tip.",
               "Check: simplified top edge stays one pixel below the crop origin."],
    )

    root = ET.fromstring(svg)
    assert root.attrib["viewBox"] == "0.00 0.00 201.00 222.00"
    polygon = root.find(f"{SVG_NS}polygon")
    points = np.array([list(map(float, point.split(","))) for point in polygon.attrib["points"].split()])
    assert len(points) == 4
    assert set(map(tuple, points)) == {(0, 1), (200, 1), (200, 221), (0, 221)}


@pytest.mark.parametrize("scale", [0.5, 1.0])
@pytest.mark.parametrize(
    "image_path, reference_size",
    [
        ("ades-light-blue-1/ades-light-blue-1.jpg", (762, 762)),
        ("iswic/iswic.jpg", (746.125, 1098.55)),
    ],
)
def test_measure_returns_cropped_photo_and_embeds_same_png_in_svg(
    image_path, reference_size, scale, svg_background_comparison,
):
    photo = cv2.imread(str(TEST_IMAGES_ROOT / image_path))
    assert photo is not None
    original = photo.copy()
    measurer = ReferenceSurfaceMeasurer(
        scale=scale, reference_size_mm=reference_size, save_debug_images=False,
    )

    measurements, debug = measurer.measure(photo, return_debug=True)
    svg_background_comparison(
        title=f"{Path(image_path).parent.name} | scale {scale:g}",
        source=photo, image_base64=measurer.image_base64, svg=measurer.svg,
        notes=[f"Corrected crop bounds (x, y, width, height): {measurements[0].bbox}"
               if measurements else "No measurement returned.",
               "Checks: exact crop pixels, unchanged input, shared PNG payload."],
    )

    assert debug["status"] == "ok"
    assert measurements
    payload = debug["object_image_base64"]
    assert measurer.image_base64 == payload
    crop = decode_png(payload)
    x, y, width, height = measurements[0].bbox
    assert crop.shape[:2] == (height, width)
    np.testing.assert_array_equal(crop, debug["imgWarp"][y:y + height, x:x + width])
    np.testing.assert_array_equal(photo, original)

    assert measurer.svg == debug["object_contour_svg"]
    root = ET.fromstring(measurer.svg)
    image, polygon = list(root)
    assert image.tag == f"{SVG_NS}image"
    assert image.attrib["href"] == f"data:image/png;base64,{payload}"
    assert float(image.attrib["width"]) == pytest.approx(measurements[0].width_mm, abs=0.005)
    assert float(image.attrib["height"]) == pytest.approx(measurements[0].height_mm, abs=0.005)
    assert polygon.tag == f"{SVG_NS}polygon"
    assert polygon.attrib["fill"] == "none"


def test_default_measure_return_exposes_photo_and_resets_it_after_failure(svg_background_comparison):
    measurer = ReferenceSurfaceMeasurer(reference_size_mm=(762, 762), save_debug_images=False)
    assert measurer.image_base64 is None
    assert measurer.svg is None
    photo = cv2.imread(str(TEST_IMAGES_ROOT / "ades-light-blue-1/ades-light-blue-1.jpg"))
    assert photo is not None

    measurements = measurer.measure(photo)
    svg_background_comparison(
        title="Measurer reuse | empty input clears both outputs",
        source=photo, image_base64=measurer.image_base64, svg=measurer.svg,
        notes=["Panels show the successful outputs before the empty-input call."],
    )

    assert measurements
    assert measurer.image_base64 is not None
    assert measurer.svg is not None
    empty_result = measurer.measure(np.empty((0, 0, 3), dtype=np.uint8))
    svg_background_comparison(notes=[
        "Panels show the successful outputs before the empty-input call.",
        f"Empty input -> {empty_result} | image_base64={measurer.image_base64} | svg={measurer.svg}",
    ])
    assert empty_result == []
    assert measurer.image_base64 is None
    assert measurer.svg is None
    assert measurer.debug["object_image_base64"] is None

