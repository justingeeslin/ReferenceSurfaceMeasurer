"""Build one visual comparison artifact from the photo tests in each run."""

from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import cv2
import numpy as np
import pytest


PANEL_W, PANEL_H = 360, 360
PANEL_PADDING = 32
GAP = 12
CARD_W = PANEL_W * 3 + GAP * 4
CARD_H = PANEL_H + 276


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    report = (yield).get_result()
    reports = item.config.__dict__.setdefault("_comparison_reports", {})
    if report.when == "call" or report.failed or report.skipped:
        # Preserve failures, including failures during teardown.
        if reports.get(item.nodeid) != "FAIL":
            reports[item.nodeid] = {"passed": "PASS", "failed": "FAIL", "skipped": "SKIP"}[report.outcome]


def aligned_expected(expected, actual):
    width, height, _ = expected
    return min([(width, height), (height, width)],
               key=lambda pair: max(abs(a - e) for a, e in zip(actual, pair)))


def accuracy_lines(expected, actual):
    if expected is None:
        return ["Expected measurements / tolerance: not specified (smoke test)."]
    width, height, tolerance = expected
    lines = [f"Expected W x H: {width:.2f} x {height:.2f} mm | tolerance: +/-{tolerance:g} mm per axis"]
    if actual is None:
        return lines + ["Actual measurements / differences: unavailable."]
    # Match the test's allowance for swapped axes; prefer a passing pairing.
    aligned = aligned_expected(expected, actual)
    differences = [a - e for a, e in zip(actual, aligned)]
    excess = [max(0, abs(delta) - tolerance) for delta in differences]
    lines += [f"Actual W x H: {actual[0]:.2f} x {actual[1]:.2f} mm (dominant contour)",
              f"Compared against W x H: {aligned[0]:.2f} x {aligned[1]:.2f} mm"
              + (" (expected axes swapped, as allowed by test)" if aligned != (width, height) else ""),
              f"Difference (actual - expected): Width {differences[0]:+.2f} mm | Height {differences[1]:+.2f} mm",
              f"Beyond tolerance: Width {excess[0]:.2f} mm | Height {excess[1]:.2f} mm"]
    return lines


def photo_panel(image):
    panel = np.full((PANEL_H, PANEL_W, 3), 245, dtype=np.uint8)
    if image is None:
        cv2.putText(panel, "Unavailable", (65, 145), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (70, 70, 70), 1, cv2.LINE_AA)
        return panel
    h, w = image.shape[:2]
    scale = min((PANEL_W - 2 * PANEL_PADDING) / w, (PANEL_H - 2 * PANEL_PADDING) / h)
    resized = cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))),
                         interpolation=cv2.INTER_AREA)
    h, w = resized.shape[:2]
    x, y = (PANEL_W - w) // 2, (PANEL_H - h) // 2
    panel[y:y + h, x:x + w] = resized
    return panel


def dimension_label(panel, text, center, vertical, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (width, height), baseline = cv2.getTextSize(text, font, 0.42, 1)
    label = np.full((height + baseline + 8, width + 10, 3), 255, dtype=np.uint8)
    cv2.putText(label, text, (5, height + 3), font, 0.42, color, 1, cv2.LINE_AA)
    if vertical:
        label = cv2.rotate(label, cv2.ROTATE_90_COUNTERCLOCKWISE)
    h, w = label.shape[:2]
    x = int(np.clip(round(center[0] - w / 2), 0, PANEL_W - w))
    y = int(np.clip(round(center[1] - h / 2), 0, PANEL_H - h))
    panel[y:y + h, x:x + w] = label


def expected_measurement_box(measurement, expected):
    width, height = aligned_expected(expected, (measurement.width_mm, measurement.height_mm))
    box = measurement.measurement_box
    center = box.mean(axis=0)
    horizontal = (box[1] - box[0]) * (width / measurement.width_mm) / 2
    vertical = (box[2] - box[1]) * (height / measurement.height_mm) / 2
    return np.array([center - horizontal - vertical, center + horizontal - vertical,
                     center + horizontal + vertical, center - horizontal + vertical])


def dashed_box(panel, box, color):
    for start, end in zip(box, np.roll(box, -1, axis=0)):
        length = np.linalg.norm(end - start)
        for distance in np.arange(0, length, 11):
            first = start + (end - start) * distance / max(length, 1)
            last = start + (end - start) * min(distance + 7, length) / max(length, 1)
            cv2.line(panel, tuple(np.rint(first).astype(int)), tuple(np.rint(last).astype(int)),
                     color, 2, cv2.LINE_AA)


def measurement_panel(outline, measurements, expected=None):
    panel = photo_panel(outline)
    h, w = outline.shape[:2]
    scale = min((PANEL_W - 2 * PANEL_PADDING) / w, (PANEL_H - 2 * PANEL_PADDING) / h)
    resized_w, resized_h = max(1, round(w * scale)), max(1, round(h * scale))
    offset = np.array([(PANEL_W - resized_w) // 2, (PANEL_H - resized_h) // 2])
    if (expected is not None and measurements and measurements[0].measurement_box is not None
            and measurements[0].width_mm > 0 and measurements[0].height_mm > 0):
        box = expected_measurement_box(measurements[0], expected)
        box = box * [resized_w / w, resized_h / h] + offset
        color = (180, 35, 175)
        dashed_box(panel, box, color)
        clipped = np.any(box < 0) or np.any(box >= [PANEL_W, PANEL_H])
        cv2.putText(panel, "Dashed: expected, centered" + (" (clipped)" if clipped else ""),
                    (8, PANEL_H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1, cv2.LINE_AA)
    for measurement in measurements:
        if measurement.measurement_box is None:
            continue
        box = measurement.measurement_box * [resized_w / w, resized_h / h] + offset
        cv2.polylines(panel, [np.rint(box).astype(np.int32)], True, (130, 130, 130), 1, cv2.LINE_AA)
        for index, (name, value, color) in enumerate((
            ("Width", measurement.width_mm, (190, 85, 25)),
            ("Height", measurement.height_mm, (25, 100, 210)),
        )):
            start, end = box[index], box[index + 1]
            for first, last in ((start, end), (end, start)):
                cv2.arrowedLine(panel, tuple(np.rint(first).astype(int)),
                                tuple(np.rint(last).astype(int)), color, 1, cv2.LINE_AA, tipLength=0.04)
            middle = (start + end) / 2
            outward = middle - box.mean(axis=0)
            outward /= max(np.linalg.norm(outward), 1)
            vertical = abs(end[1] - start[1]) > abs(end[0] - start[0])
            dimension_label(panel, f"{name} {value:.1f} mm", middle + outward * 15, vertical, color)
    return panel


@pytest.fixture(scope="session")
def _contour_comparison(request):
    # Store thumbnails only, and deduplicate cases shared by smoke/measurement tests.
    cards = {}
    cases = {}

    def record(nodeid, expected, slug, scale, image, measurements, debug):
        key = (slug, scale)
        cases.setdefault(key, {})[nodeid] = (expected, (measurements[0].width_mm, measurements[0].height_mm)
                                           if measurements else None)
        if image is None and key in cards:
            return
        panel_expected = next((value[0] for value in cases[key].values() if value[0] is not None), None)
        warp = debug.get("imgWarp")
        outline = None
        if warp is not None:
            outline = np.full_like(warp, 255)
            contours = [m.contour for m in measurements if m.contour is not None]
            cv2.drawContours(outline, contours, -1, (40, 145, 20),
                             max(2, round(min(warp.shape[:2]) / 140)), cv2.LINE_AA)
        card = np.full((CARD_H, CARD_W, 3), 255, dtype=np.uint8)
        title = f"{slug} | scale {scale} | {len(measurements)} contour(s)"
        if image is None:
            title += " | PHOTO UNAVAILABLE"
        elif debug.get("status") != "ok":
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
            card[66:66 + PANEL_H, x:x + PANEL_W] = (
                measurement_panel(photo, measurements, panel_expected)
                if index == 2 and photo is not None else photo_panel(photo)
            )
        cards[key] = card

    yield record

    if not cards:
        return
    reports = request.config.__dict__.get("_comparison_reports", {})
    for key, card in cards.items():
        results = cases[key]
        outcomes = [reports.get(nodeid, "UNKNOWN") for nodeid in results]
        outcome = "FAIL" if "FAIL" in outcomes else ("PASS" if all(o == "PASS" for o in outcomes) else "/".join(sorted(set(outcomes))))
        smoke = [reports.get(nodeid, "UNKNOWN") for nodeid, (expected, _) in results.items() if expected is None]
        measured = [reports.get(nodeid, "UNKNOWN") for nodeid, (expected, _) in results.items() if expected is not None]
        detail = " | ".join(f"{name}: {'/'.join(sorted(set(values)))} ({len(values)} test(s))"
                            for name, values in (("Measurement", measured), ("Smoke", smoke)) if values)
        color = (35, 130, 35) if outcome == "PASS" else (30, 40, 190)
        cv2.putText(card, f"Tests: {outcome} | {detail}", (GAP, PANEL_H + 92),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)
        expected, actual = next((value for value in results.values() if value[0] is not None), next(iter(results.values())))
        lines = accuracy_lines(expected, actual)
        if expected is not None:
            lines.append("Accuracy assertion checks the dominant contour; contour count is not asserted.")
        if actual is None:
            lines.append("Photo or measurable contour unavailable; see pytest output for the failure.")
        for index, line in enumerate(lines):
            cv2.putText(card, line, (GAP, PANEL_H + 120 + index * 23),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 50, 50), 1, cv2.LINE_AA)
    rows = (len(cards) + 1) // 2
    sheet = np.full((rows * (CARD_H + GAP) + 100, CARD_W * 2 + GAP * 3, 3),
                    225, dtype=np.uint8)
    cv2.putText(sheet, "Photo / extracted contour comparison", (GAP, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (35, 35, 35), 2, cv2.LINE_AA)
    cv2.putText(sheet, f"{len(cards)} photo/scale cases from this test run", (GAP, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (65, 65, 65), 1, cv2.LINE_AA)
    timestamp = datetime.now(ZoneInfo("America/Chicago")).strftime("%Y-%m-%d %H:%M:%S %Z")
    summary = " | ".join(f"{sum(value == status for value in reports.values())} {status}" for status in ("PASS", "FAIL", "SKIP"))
    cv2.putText(sheet, f"Run: {timestamp} | Full selected suite: {summary}", (GAP, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (65, 65, 65), 1, cv2.LINE_AA)
    for index, key in enumerate(sorted(cards)):
        x = GAP + (index % 2) * (CARD_W + GAP)
        y = 100 + (index // 2) * (CARD_H + GAP)
        sheet[y:y + CARD_H, x:x + CARD_W] = cards[key]

    path = Path(request.config.rootpath) / "assets" / "contour-comparison.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise OSError(f"Could not write contour comparison: {path}")
    reporter = request.config.pluginmanager.get_plugin("terminalreporter")
    if reporter:
        reporter.write_line(f"Contour comparison: {path}")


@pytest.fixture
def contour_comparison(request, _contour_comparison):
    params = request.node.callspec.params
    expected = ((params["expected_w_mm"], params["expected_h_mm"], params["tol_mm"])
                if "expected_w_mm" in params else None)

    def record(slug, scale, image, measurements, debug):
        _contour_comparison(request.node.nodeid, expected, slug, scale, image, measurements, debug)

    # Include failed photo loads too, even when the test never reaches record().
    record(params["slug"], params["scale"], None, [], {})
    return record
