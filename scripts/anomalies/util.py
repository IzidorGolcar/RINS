import cv2
import numpy as np
from pathlib import Path


def generate_tile_mask(
    image: np.ndarray,
    morph_close_size: int = 25,
    morph_open_size: int = 10,
) -> tuple[np.ndarray, np.ndarray]:

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
 
    close_k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (morph_close_size, morph_close_size)
    )
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_k)
 
    open_k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (morph_open_size, morph_open_size)
    )
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, open_k)
 
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        opened, connectivity=8
    )
    if num_labels < 2:
        return np.zeros(image.shape[:2], dtype=np.uint8), None
 
    largest_label = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    blob = np.where(labels == largest_label, 255, 0).astype(np.uint8)
 
    contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv2.contourArea)
 
    hull = cv2.convexHull(cnt)
    peri = cv2.arcLength(hull, True)
    quad = cv2.approxPolyDP(hull, 0.05 * peri, True)
 
    if len(quad) != 4:
        rect = cv2.minAreaRect(hull)
        quad = cv2.boxPoints(rect).astype(np.int32).reshape(-1, 1, 2)
 
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [quad], 255)
    return mask, quad


def is_tile_fully_visible(
        quad: np.ndarray,
        image_shape: tuple,
        border_threshold: int = 10
    ) -> bool:
    h, w = image_shape[:2]
    pts = quad.reshape(4, 2)
    return bool(
        np.all(pts[:, 0] > border_threshold) and
        np.all(pts[:, 0] < w - border_threshold) and
        np.all(pts[:, 1] > border_threshold) and
        np.all(pts[:, 1] < h - border_threshold)
    )
 



def rectify_tile(
    image: np.ndarray,
    quad: np.ndarray,
    output_size: int = 512,
) -> np.ndarray:
    pts = quad.reshape(4, 2).astype(np.float32)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    pts = pts[np.argsort(angles)]
    pts = np.roll(pts, -np.argmin(pts.sum(axis=1)), axis=0)
    s = output_size - 1
    dst = np.array([[0, 0], [s, 0], [s, s], [0, s]], dtype=np.float32)
    H, _ = cv2.findHomography(pts, dst)
    warped = cv2.warpPerspective(image, H, (output_size, output_size))
    return warped
