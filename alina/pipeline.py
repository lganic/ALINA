from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime

import cv2
import numpy as np

from .color import normalize_color_features
from .config import PipelineConfig
from .histogram import calc_histogram
from .roi import roi_points_to_quad
from .traversal import circledat_fast, circular_threshold_pixel_discovery_and_traversal


@dataclass
class ImageResult:
    filename: str
    had_lines: bool
    num_pixels: int
    elapsed_seconds: float


def process_image(img: np.ndarray, roi_points: np.ndarray, config: PipelineConfig) -> tuple[np.ndarray, np.ndarray, bool]:
    final_img = img.copy()
    no_lines_img = img.copy()

    # 3.3 perspective transformation -> bird's eye view of the ROI
    roi_quad = roi_points_to_quad(roi_points, dtype=np.float32)
    dst = np.array(
        [
            config.roi.dst_bottom_left,
            config.roi.dst_top_left,
            config.roi.dst_top_right,
            config.roi.dst_bottom_right,
        ],
        dtype=np.float32,
    ).reshape(1, 4, 2)

    M = cv2.getPerspectiveTransform(roi_quad, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))

    # 3.4 color feature normalization (HSV, per-channel min-max)
    color_features = normalize_color_features(warped)

    # 3.5 HSV-based color thresholding -> binary mask of candidate line pixels
    lower = np.array(config.yellow_lower)
    upper = np.array(config.yellow_upper)
    mask = cv2.inRange(color_features, lower, upper)
    if config.mask_ignore_left_columns > 0:
        mask[:, :config.mask_ignore_left_columns] = 0

    # 3.6 histogram analysis -> vertical projection, peak column
    avg_pixel, peak_value = calc_histogram(mask, min_white_pixels=config.min_white_pixels)

    # 3.7 threshold check -> is this peak an actual line marking or noise
    if not avg_pixel or peak_value <= config.peak_pixel_threshold:
        return no_lines_img, np.empty((0, 2), dtype=np.int32), False

    # 3.8 CIRCLEDAT -> traverse from the peak to collect all connected line pixels
    binary = mask.copy()
    binary[avg_pixel[1]][avg_pixel[0]] = 255
    line_marking_pixels = circular_threshold_pixel_discovery_and_traversal(
        binary, avg_pixel[0], avg_pixel[1], config.circular_threshold
    )

    height, width = binary.shape[:2]
    black_img = np.zeros((height, width), dtype=np.uint8)
    if config.fast_traversal:
        pts = circledat_fast(binary, avg_pixel[0], avg_pixel[1], config.circular_threshold)
        black_img[pts[:, 1], pts[:, 0]] = 255
        # same footprint as cv2.circle(radius=1, filled): a 3x3 cross
        black_img = cv2.dilate(black_img, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)))
    else:
        line_marking_pixels = circular_threshold_pixel_discovery_and_traversal(
            binary, avg_pixel[0], avg_pixel[1], config.circular_threshold
        )
        for x, y in line_marking_pixels:
            cv2.circle(black_img, (x, y), 1, 255, -1)

    # 3.9 frame unwarping -> map detected pixels back to the original perspective
    Minv = cv2.getPerspectiveTransform(dst, roi_quad)
    unwarped = cv2.warpPerspective(black_img, Minv, (img.shape[1], img.shape[0]))

    line_pixels_only = np.where(unwarped > 0)
    x_coords, y_coords = line_pixels_only[1], line_pixels_only[0]
    coords = np.column_stack((x_coords, y_coords))

    # 3.9 annotate -> mark line marking pixels in red on the original frame
    final_img[line_pixels_only] = (0, 0, 255)
    return final_img, coords, True


def run_batch(config: PipelineConfig, roi_points: np.ndarray) -> list[ImageResult]:
    image_filenames = sorted(
        f for f in config.input_dir.iterdir() if f.suffix.lower() == ".jpg"
    )
    total = len(image_filenames)

    # confirm the ROI on the reference frame before starting the batch
    reference_quad = roi_points_to_quad(roi_points, dtype=np.int32)
    reference_image = cv2.imread(str(image_filenames[0]))

    from .roi import compute_display_scale
    scale = compute_display_scale(reference_image, max_display_dim=1080)
    display_image = cv2.resize(reference_image, None, fx=scale, fy=scale) if scale != 1.0 else reference_image
    display_quad = (reference_quad * scale).astype(np.int32) if scale != 1.0 else reference_quad

    cv2.polylines(display_image, display_quad, True, (0, 0, 255), thickness=2)
    cv2.imshow("Confirm ROI", display_image)
    cv2.waitKey()
    cv2.destroyAllWindows()

    print("Labeling Process Initiated!")

    results: list[ImageResult] = []
    log_file = open(config.log_file, "w") if config.log_file else None

    try:
        for i, path in enumerate(image_filenames, start=1):
            start = time.perf_counter()
            img = cv2.imread(str(path))
            if img is None:
                continue

            annotated, coords, had_lines = process_image(img, roi_points, config)
            elapsed = time.perf_counter() - start

            text_filename = path.stem + ".txt"
            np.savetxt(config.output_coords_dir / text_filename, coords, fmt="%6d")
            cv2.imwrite(str(config.output_images_dir / path.name), annotated)

            status = "Labeled" if had_lines else "No lines found"
            log_line = (
                f"[{i}/{total}] [{status}] {path.name}: {len(coords)} px, "
                f"elapsed {elapsed:.3f}s ({datetime.now().strftime('%H:%M:%S')})"
            )
            print(log_line)

            if log_file:
                log_file.write(log_line + "\n")
                log_file.flush()

            results.append(ImageResult(path.name, had_lines, len(coords), elapsed))
    finally:
        if log_file:
            log_file.close()

    labeled_count = sum(1 for r in results if r.had_lines)
    total_elapsed = sum(r.elapsed_seconds for r in results)
    avg_ms = (total_elapsed / len(results) * 1000) if results else 0
    fps = (1000 / avg_ms) if avg_ms else 0
    summary = (
        f"Done: {labeled_count}/{len(results)} labeled, "
        f"avg {avg_ms:.1f}ms/frame ({fps:.2f} fps), total {total_elapsed:.1f}s"
    )
    print(summary)

    if config.log_file:
        with open(config.log_file, "a") as f:
            f.write(summary + "\n")

    return results