from __future__ import annotations

from itertools import product

import cv2
import numpy as np


def circular_threshold_pixel_discovery_and_traversal(img: np.ndarray, x: int, y: int, threshold: int) -> list[tuple[int, int]]:
    visited = {(x, y)}
    line_marking_pixels = []
    stack = [(x, y)]

    circular_range = list(product(range(-threshold, threshold + 1), repeat=2))
    circular_range.remove((0, 0))

    height, width = img.shape[:2]

    while stack:
        cx, cy = stack.pop()

        if img[cy][cx] != 255:
            continue

        line_marking_pixels.append((cx, cy))

        for dx, dy in circular_range:
            nx, ny = cx + dx, cy + dy
            if (nx, ny) in visited:
                continue
            if nx < 0 or ny < 0 or nx >= width or ny >= height:
                continue
            stack.append((nx, ny))
            visited.add((nx, ny))

    return line_marking_pixels


def circledat_fast(img: np.ndarray, x: int, y: int, threshold: int) -> np.ndarray:
    """Same pixel set as circular_threshold_pixel_discovery_and_traversal, in O(pixels).

    The traversal links two line pixels when both |dx| and |dy| are <= threshold. Dilating every
    line pixel into a (threshold+1)x(threshold+1) square makes two squares 8-adjacent or
    overlapping under exactly that condition, so the 8-connected component of the dilated
    mask that contains the seed holds exactly the pixels the traversal reaches.
    Returns an (N, 2) array of (x, y).
    """
    binary = (img == 255).astype(np.uint8)
    if not binary[y, x]:
        return np.empty((0, 2), dtype=np.int64)
    # squares of side t (same anchor for every pixel) touch or overlap iff |dx| <= t and |dy| <= t
    kernel = np.ones((threshold, threshold), np.uint8)
    anchor = (threshold // 2, threshold // 2)
    grown = cv2.dilate(binary, kernel, anchor=anchor)
    _, labels = cv2.connectedComponents(grown, connectivity=8)
    ys, xs = np.nonzero((labels == labels[y, x]) & (binary > 0))
    return np.column_stack((xs, ys))
