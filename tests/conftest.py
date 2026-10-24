"""Build one visual comparison artifact from the photo tests in each run."""

from pathlib import Path

import cv2
import numpy as np
import pytest


PANEL_W, PANEL_H = 280, 280
GAP = 12
CARD_W = PANEL_W * 3 + GAP * 4
CARD_H = PANEL_H + 86


def photo_panel(image):
    panel = np.full((PANEL_H, PANEL_W, 3), 245, dtype=np.uint8)
    if image is None:
        cv2.putText(panel, "Unavailable", (65, 145), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (70, 70, 70), 1, cv2.LINE_AA)
        return panel
    h, w = image.shape[:2]
    scale = min(PANEL_W / w, PANEL_H / h)
    resized = cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))),
                         interpolation=cv2.INTER_AREA)
    h, w = resized.shape[:2]
    x, y = (PANEL_W - w) // 2, (PANEL_H - h) // 2
    panel[y:y + h, x:x + w] = resized
    return panel


@pytest.fixture(scope="session")
def contour_comparison(request):
    # Store thumbnails only, and deduplicate cases shared by smoke/measurement tests.
    cards = {}

    def record(slug, scale, image, measurements, debug):
        warp = debug.get("imgWarp")
        outline = None
        if warp is not None:
            outline = np.full_like(warp, 255)
            contours = [m.contour for m in measurements if m.contour is not None]
            cv2.drawContours(outline, contours, -1, (40, 145, 20),
                             max(2, round(min(warp.shape[:2]) / 140)), cv2.LINE_AA)
        card = np.full((CARD_H, CARD_W, 3), 255, dtype=np.uint8)
        title = f"{slug} | scale {scale} | {len(measurements)} contour(s)"
        if debug.get("status") != "ok":
            title += " | EXTRACTION FAILED"
        cv2.putText(card, title, (GAP, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.47, (35, 35, 35), 1, cv2.LINE_AA)
        for index, (label, photo) in enumerate((
            ("Original photo", image),
            ("Perspective-corrected photo", warp),
            ("Extracted contour (corrected view)", outline),
        )):
            x = GAP + index * (PANEL_W + GAP)
            cv2.putText(card, label, (x, 53), cv2.FONT_HERSHEY_SIMPLEX,
                        0.43, (65, 65, 65), 1, cv2.LINE_AA)
            card[66:66 + PANEL_H, x:x + PANEL_W] = photo_panel(photo)
        cards[(slug, scale)] = card

    yield record

    if not cards:
        return
    rows = (len(cards) + 1) // 2
    sheet = np.full((rows * (CARD_H + GAP) + 70, CARD_W * 2 + GAP * 3, 3),
                    225, dtype=np.uint8)
    cv2.putText(sheet, "Photo / extracted contour comparison", (GAP, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (35, 35, 35), 2, cv2.LINE_AA)
    cv2.putText(sheet, f"{len(cards)} photo/scale cases from this test run", (GAP, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (65, 65, 65), 1, cv2.LINE_AA)
    for index, key in enumerate(sorted(cards)):
        x = GAP + (index % 2) * (CARD_W + GAP)
        y = 70 + (index // 2) * (CARD_H + GAP)
        sheet[y:y + CARD_H, x:x + CARD_W] = cards[key]

    path = Path(request.config.rootpath) / "build" / "test-results" / "contour-comparison.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise OSError(f"Could not write contour comparison: {path}")
    reporter = request.config.pluginmanager.get_plugin("terminalreporter")
    if reporter:
        reporter.write_line(f"Contour comparison: {path}")
