#!/usr/bin/python3
"""Barrel detector for Task 2 (Industry 5.0).

Locates barrels, recognises colour, classifies orientation (vertical /
horizontal), and detects spills under horizontal barrels.  Fuses an
OAK-D forward stream with a top_camera downward stream into a single
shared landmark map; spills must be seen by both cameras (or several
times by one) before they latch as confirmed.

See the design plan for rationale; key reuse from ``detect_rings.py``
and ``ring_map.py``.
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile,
                       QoSReliabilityPolicy)
from rclpy.time import Time as RclTime
from rclpy.duration import Duration

import tf2_ros
import tf2_geometry_msgs  # registers PointStamped transform support  # noqa: F401

from cv_bridge import CvBridge
import message_filters

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Vector3, Quaternion
from std_msgs.msg import ColorRGBA, String
from visualization_msgs.msg import Marker, MarkerArray

# Same-package sibling import.
sys.path.insert(0, os.path.dirname(__file__))
from barrel_map import BarrelMap, BarrelLandmark  # noqa: E402


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

# HSV colour gates for barrel surfaces.
# Each entry is a list of (H_lo, S_lo, V_lo, H_hi, S_hi, V_hi) ranges that
# get OR-ed together.  Red wraps the hue circle so it has two ranges.
COLOUR_RANGES: dict[str, list[tuple[int, int, int, int, int, int]]] = {
    'red':    [(0, 120, 80, 10, 255, 255), (170, 120, 80, 180, 255, 255)],
    'orange': [(11, 130, 110, 22, 255, 255)],
    'yellow': [(23, 120, 120, 35, 255, 255)],
    'green':  [(40, 100, 50, 85, 255, 255)],
    'blue':   [(95, 120, 50, 130, 255, 255)],
    'purple': [(131, 100, 50, 160, 255, 255)],
    'brown':  [(0, 80, 30, 22, 200, 110)],
    'black':  [(0, 0, 0, 180, 255, 80)],
}

# RGB used for the RViz CYLINDER marker colour swatch.
COLOUR_RGB: dict[str, tuple[float, float, float]] = {
    'red':    (1.0, 0.0, 0.0),
    'orange': (1.0, 0.55, 0.0),
    'yellow': (1.0, 1.0, 0.0),
    'green':  (0.0, 0.8, 0.0),
    'blue':   (0.0, 0.3, 1.0),
    'purple': (0.6, 0.0, 0.8),
    'brown':  (0.45, 0.27, 0.07),
    'black':  (0.15, 0.15, 0.15),
    'unknown': (0.5, 0.5, 0.5),
}

# Physical sanity bounds (standard 200 L barrel ≈ 0.57 m d × 0.88 m h).
BARREL_RADIUS_RANGE = (0.10, 0.45)   # m
BARREL_LENGTH_RANGE = (0.25, 1.25)     # m

# PCA classification thresholds.  A real barrel surface seen from one side
# gives λ0/λ1 ≈ 3–4 (length variance / circle-half variance).  The cosine-Z
# check below is what actually distinguishes vertical from horizontal —
# the ratio gate is just there to reject non-elongated colour blobs.
PCA_AXIS_RATIO_VERT = 1.0
PCA_AXIS_RATIO_HORIZ = 1.0
VERTICAL_COS_MIN = 0.85          # dot(v0, world_z)
HORIZONTAL_COS_MAX = 0.30

# Per-blob filters.
MIN_BLOB_POINTS = 50

# Spill detection.
SPILL_SEARCH_PIXEL_PADDING = 60    # extend below barrel bbox by this many px
SPILL_MIN_AREA_M2 = 0.04           # 0.04 m² (~20 cm × 20 cm)
SPILL_MAX_DIST_TO_BARREL = 1.2     # m, from barrel ground anchor
SPILL_FLOOR_Z_MAX = 0.10           # m above map-frame floor
SPILL_FLOOR_DEPTH_VAR = 0.015      # m std-dev of fitted plane

# Marker IDs are partitioned by namespace so they never collide.
NS_BARREL = 'confirmed_barrels'
NS_LABEL = 'confirmed_barrels_labels'
NS_SPILL = 'confirmed_barrels_spill'

_QOS_LATCHED = QoSProfile(
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)


# ---------------------------------------------------------------------------
# Per-camera configuration
# ---------------------------------------------------------------------------

@dataclass
class CameraCfg:
    key: str
    topic_prefix: str
    depth_min: float
    depth_max: float
    min_contour_area: int
    discovery: bool       # may create new barrel landmarks?
    spill_search: bool    # may search for spills?


CAMERAS = [
    CameraCfg(key='oakd',
              topic_prefix='/oakd/rgb/preview',
              depth_min=0.2, depth_max=4.0,
              min_contour_area=200,
              discovery=True, spill_search=True),
    CameraCfg(key='top',
              topic_prefix='/top_camera/rgb/preview',
              depth_min=0.1, depth_max=2.0,
              min_contour_area=500,
              discovery=True, spill_search=True),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def quat_to_mat(x: float, y: float, z: float, w: float) -> np.ndarray:
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def yaw_to_quat(yaw: float) -> Quaternion:
    """Quaternion (z-axis rotation only)."""
    return Quaternion(x=0.0, y=0.0, z=math.sin(yaw / 2.0), w=math.cos(yaw / 2.0))


def axis_to_quat(axis: np.ndarray) -> Quaternion:
    """Quaternion that rotates +Z to ``axis`` (axis assumed unit-norm).

    Used so the CYLINDER marker's intrinsic Z-axis lines up with the
    landmark's measured orientation axis.
    """
    z = np.array([0.0, 0.0, 1.0])
    a = axis / (np.linalg.norm(axis) + 1e-9)
    dot = float(np.clip(np.dot(z, a), -1.0, 1.0))
    if dot > 0.999999:
        return Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
    if dot < -0.999999:
        # 180° about any perpendicular axis — pick X.
        return Quaternion(x=1.0, y=0.0, z=0.0, w=0.0)
    cross = np.cross(z, a)
    s = math.sqrt(2.0 * (1.0 + dot))
    inv_s = 1.0 / s
    return Quaternion(
        x=float(cross[0] * inv_s),
        y=float(cross[1] * inv_s),
        z=float(cross[2] * inv_s),
        w=float(s * 0.5),
    )


# ---------------------------------------------------------------------------
# Detector node
# ---------------------------------------------------------------------------

class BarrelDetector(Node):
    DETECTION_EVERY_N_FRAMES = 3   # run detection only every Nth frame to save CPU

    def __init__(self) -> None:
        super().__init__('detect_barrels')

        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.barrel_map = BarrelMap()
        self._last_publish_hash: Optional[int] = None

        # Per-camera state.
        self._intrinsics: dict[str, dict] = {}      # key -> {fx, fy, cx, cy}
        self._cam_cfg: dict[str, CameraCfg] = {c.key: c for c in CAMERAS}
        self._frame_counts = {'oakd': 0, 'top': 0}
        self._last_vis = {'oakd': None, 'top': None}

        # Publishers
        self.markers_pub = self.create_publisher(MarkerArray, '/barrel_markers', _QOS_LATCHED)
        self.inspections_pub = self.create_publisher(String, '/barrel_inspections', _QOS_LATCHED)

        # Subscribe per camera.
        for cfg in CAMERAS:
            self._setup_camera(cfg)

        # Debug windows: one per camera so we can see what each pipeline sees.
        cv2.namedWindow('barrels_oakd', cv2.WINDOW_NORMAL)
        cv2.namedWindow('barrels_top', cv2.WINDOW_NORMAL)

        self.get_logger().info('Barrel detector ready (OAK-D + top_camera).')

    # ------------------------------------------------------------------ setup

    def _setup_camera(self, cfg: CameraCfg) -> None:
        rgb_sub = message_filters.Subscriber(self, Image, f'{cfg.topic_prefix}/image_raw')
        depth_sub = message_filters.Subscriber(self, Image, f'{cfg.topic_prefix}/depth')
        sync = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub], queue_size=1, slop=0.1,
        )
        sync.registerCallback(lambda rgb, depth, k=cfg.key: self._stream_cb(rgb, depth, k))

        # Keep refs alive on self so they don't get GC'd.
        setattr(self, f'_rgb_sub_{cfg.key}', rgb_sub)
        setattr(self, f'_depth_sub_{cfg.key}', depth_sub)
        setattr(self, f'_sync_{cfg.key}', sync)

        self.create_subscription(
            CameraInfo,
            f'{cfg.topic_prefix}/camera_info',
            lambda msg, k=cfg.key: self._cam_info_cb(msg, k),
            _QOS_LATCHED,
        )

    # ------------------------------------------------------------ subscribers

    def _cam_info_cb(self, msg: CameraInfo, cam_key: str) -> None:
        if cam_key in self._intrinsics:
            return
        self._intrinsics[cam_key] = {
            'fx': msg.k[0], 'fy': msg.k[4],
            'cx': msg.k[2], 'cy': msg.k[5],
        }
        self.get_logger().info(
            f'[{cam_key}] intrinsics fx={msg.k[0]:.1f} fy={msg.k[4]:.1f} '
            f'cx={msg.k[2]:.1f} cy={msg.k[5]:.1f}')

    def _stream_cb(self, rgb_msg: Image, depth_msg: Image, cam_key: str) -> None:
        if cam_key not in self._intrinsics:
            return

        self._frame_counts[cam_key] += 1
        if self._frame_counts[cam_key] % self.DETECTION_EVERY_N_FRAMES != 0:
            last = self._last_vis.get(cam_key)
            if last is not None:
                wname = 'barrels_oakd' if cam_key == 'oakd' else 'barrels_top'
                cv2.imshow(wname, last)
                cv2.waitKey(1)
            return

        try:
            rgb = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
            depth = self.bridge.imgmsg_to_cv2(depth_msg, '32FC1')
        except Exception as e:
            self.get_logger().error(f'[{cam_key}] cv_bridge: {e}')
            return

        frame_id = rgb_msg.header.frame_id
        self.process_frame(rgb, depth, frame_id, cam_key)
        cv2.waitKey(1)

    # ------------------------------------------------------------ main pipeline

    def process_frame(self, rgb: np.ndarray, depth: np.ndarray,
                      frame_id: str, cam_key: str) -> None:
        cfg = self._cam_cfg[cam_key]

        h, w = rgb.shape[:2]
        # Depth ROI mask.
        finite = np.isfinite(depth)
        in_range = (depth > cfg.depth_min) & (depth < cfg.depth_max) & finite

        if not np.any(in_range):
            self._show_debug(cam_key, rgb, [])
            return

        hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)

        # Per-colour masks, processed in priority order to suppress brown/red overlap.
        priority = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'brown', 'black']
        consumed = np.zeros((h, w), dtype=bool)
        candidates: list[dict] = []

        for colour in priority:
            mask = np.zeros((h, w), dtype=np.uint8)
            for h_lo, s_lo, v_lo, h_hi, s_hi, v_hi in COLOUR_RANGES[colour]:
                mask |= cv2.inRange(hsv,
                                    np.array([h_lo, s_lo, v_lo]),
                                    np.array([h_hi, s_hi, v_hi]))
            # Apply depth + already-consumed gates.
            mask[~in_range] = 0
            mask[consumed] = 0
            if not mask.any():
                continue

            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < cfg.min_contour_area:
                    continue
                bbox = cv2.boundingRect(cnt)
                # Build per-blob mask.
                blob_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(blob_mask, [cnt], -1, 255, thickness=cv2.FILLED)
                blob_bool = blob_mask.astype(bool) & in_range
                if int(blob_bool.sum()) < MIN_BLOB_POINTS:
                    continue

                candidates.append({
                    'colour': colour,
                    'contour': cnt,
                    'bbox': bbox,
                    'mask': blob_bool,
                })
                consumed |= blob_bool

        # Process each candidate.
        results = []
        for cand in candidates:
            res = self._classify_candidate(cand, depth, frame_id, cam_key, cfg)
            if res is not None:
                results.append(res)

        self._show_debug(cam_key, rgb, results)

        # Spill search for any horizontal barrel observed this frame.
        for res in results:
            if res['orientation'] != 'horizontal':
                continue
            if not cfg.spill_search:
                continue
            lm = res['landmark']
            spill_map = self._detect_spill(rgb, depth, frame_id, cam_key, cfg,
                                           res['bbox'], lm.ground_anchor, res['colour'])
            if spill_map is not None:
                self.barrel_map.register_spill(lm, spill_map, cam_key)

        # Publish if confirmed set changed.
        self._publish()

    # --------------------------------------------------------- per-candidate

    def _classify_candidate(self, cand: dict, depth: np.ndarray,
                            frame_id: str, cam_key: str, cfg: CameraCfg,
                            ) -> Optional[dict]:
        K = self._intrinsics[cam_key]
        mask = cand['mask']

        ys, xs = np.where(mask)
        zs = depth[ys, xs]
        valid = np.isfinite(zs) & (zs > cfg.depth_min) & (zs < cfg.depth_max)
        ys = ys[valid]
        xs = xs[valid]
        zs = zs[valid]
        if zs.size < MIN_BLOB_POINTS:
            return None

        # Backproject to optical-frame 3D points.
        X = (xs - K['cx']) * zs / K['fx']
        Y = (ys - K['cy']) * zs / K['fy']
        pts_opt = np.stack([X, Y, zs], axis=1)

        # Optical → body axes (REP-103): body.x=opt.z, body.y=-opt.x, body.z=-opt.y.
        pts_body = np.stack([pts_opt[:, 2], -pts_opt[:, 0], -pts_opt[:, 1]], axis=1)

        # Transform to map frame.
        pts_map = self._transform_points(pts_body, frame_id)
        if pts_map is None:
            return None

        # Filter out points that are flat on the floor (potential spill points)
        # to prevent them from distorting the barrel shape during PCA/dimensions check.
        above_floor = pts_map[:, 2] > 0.08
        if above_floor.sum() < MIN_BLOB_POINTS:
            self.get_logger().info(f"-> Rejected {cand['colour']}: Too few points ({above_floor.sum()}) above floor level (>0.08m)")
            return None
        # Ground anchor (lowest point of the barrel above the floor)
        ground_anchor = pts_map[np.argmin(pts_map[:, 2])]

        # Filter out distant background structures (like tunnel walls) by keeping only points within 1.1m of the ground anchor
        dist_to_anchor = np.linalg.norm(pts_map - ground_anchor, axis=1)
        pts_map = pts_map[dist_to_anchor < 1.1]
        if pts_map.shape[0] < MIN_BLOB_POINTS:
            self.get_logger().info(f"-> Rejected {cand['colour']}: Too few points ({pts_map.shape[0]}) within 1.1m of ground anchor")
            return None

        # PCA on map-frame point cloud (orientation reasoned in world frame).
        centroid = pts_map.mean(axis=0)
        cov = np.cov((pts_map - centroid).T)
        try:
            eigvals, eigvecs = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            return None
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        v0 = eigvecs[:, 0]
        lam0, lam1 = max(eigvals[0], 1e-9), max(eigvals[1], 1e-9)

        cos_z = abs(float(np.dot(v0, np.array([0.0, 0.0, 1.0]))))
        ratio = lam0 / lam1

        # Calculate Z-extent
        z_extent = float(pts_map[:, 2].max() - pts_map[:, 2].min())

        # Ground anchor
        ground_anchor = pts_map[np.argmin(pts_map[:, 2])]

        # Length: extent along v0
        proj = (pts_map - centroid) @ v0
        length = float(proj.max() - proj.min())

        # Radius: extent perpendicular to v0
        perp = pts_map - centroid - np.outer(proj, v0)
        radii = np.linalg.norm(perp, axis=1)
        radius_est = float(np.percentile(radii, 90))

        # Orientation candidate
        orientation: Optional[str] = None
        if cos_z > VERTICAL_COS_MIN and ratio > PCA_AXIS_RATIO_VERT:
            orientation = 'vertical'
        elif cos_z < HORIZONTAL_COS_MAX and ratio > PCA_AXIS_RATIO_HORIZ:
            orientation = 'horizontal'

        # Log measured stats for the candidate
        self.get_logger().info(
            f"[MEASUREMENT] Color: {cand['colour']}, Centroid: [{centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f}], "
            f"Ratio: {ratio:.2f}, Cos_Z: {cos_z:.2f}, Length: {length:.2f}, Radius: {radius_est:.2f}, "
            f"Z-extent: {z_extent:.2f}, Ground_Z: {ground_anchor[2]:.2f}"
        )

        # Reject extremely elongated objects like floor lines
        if ratio > 6.5:
            self.get_logger().info(f"-> Rejected {cand['colour']}: Ratio {ratio:.2f} > 6.5 (too elongated)")
            return None

        if orientation is None:
            self.get_logger().info(f"-> Rejected {cand['colour']}: Ambiguous orientation (cos_z={cos_z:.2f}, ratio={ratio:.2f})")
            return None

        # Height / Z-extent sanity check (barrel length ≈ 0.88m, diameter ≈ 0.56m)
        if orientation == 'vertical' and not (0.25 <= z_extent <= 1.15):
            self.get_logger().info(f"-> Rejected {cand['colour']}: Vertical Z-extent {z_extent:.2f} out of [0.25, 1.15]")
            return None
        if orientation == 'horizontal' and not (0.20 <= z_extent <= 0.75):
            self.get_logger().info(f"-> Rejected {cand['colour']}: Horizontal Z-extent {z_extent:.2f} out of [0.20, 0.75]")
            return None

        # Ensure the barrel is resting on the floor (reject high ring stand and conveyor)
        if not (-0.15 <= ground_anchor[2] <= 0.12):
            self.get_logger().info(f"-> Rejected {cand['colour']}: Ground anchor Z {ground_anchor[2]:.2f} out of [-0.15, 0.12]")
            return None

        # Restrict to first room only (X < 4.5)
        if centroid[0] > 4.5:
            self.get_logger().info(f"-> Rejected {cand['colour']}: Centroid X {centroid[0]:.2f} > 4.5 (not first room)")
            return None

        # Size check
        if not (BARREL_LENGTH_RANGE[0] <= length <= BARREL_LENGTH_RANGE[1]):
            self.get_logger().info(f"-> Rejected {cand['colour']}: Length {length:.2f} out of range {BARREL_LENGTH_RANGE}")
            return None
        if not (BARREL_RADIUS_RANGE[0] <= radius_est <= BARREL_RADIUS_RANGE[1]):
            self.get_logger().info(f"-> Rejected {cand['colour']}: Radius {radius_est:.2f} out of range {BARREL_RADIUS_RANGE}")
            return None

        self.get_logger().info(f"-> ACCEPTED {cand['colour']} barrel!")

        # Discovery vs update.
        if not cfg.discovery:
            nearest = self.barrel_map.nearest_landmark(centroid, cand['colour'], max_dist=1.0)
            if nearest is None:
                return None

        lm = self.barrel_map.update(centroid, cand['colour'], orientation, ground_anchor, camera_id=cam_key)
        if lm is None:
            return None

        return {
            'colour': cand['colour'],
            'orientation': orientation,
            'bbox': cand['bbox'],
            'landmark': lm,
            'axis': v0,
        }

    # --------------------------------------------------------------- spills

    def _detect_spill(self, rgb: np.ndarray, depth: np.ndarray,
                      frame_id: str, cam_key: str, cfg: CameraCfg,
                      bbox: tuple[int, int, int, int],
                      barrel_anchor: Optional[np.ndarray],
                      barrel_colour: str) -> Optional[np.ndarray]:
        """Look for a saturated coloured patch on the floor near the barrel.

        Returns the map-frame centroid of the spill, or None.
        """
        if barrel_anchor is None:
            return None
        h, w = rgb.shape[:2]
        x, y, bw, bh = bbox
        # Search ROI: below the bbox, extending sideways.
        y0 = max(0, y + bh // 2)
        y1 = min(h, y + bh + SPILL_SEARCH_PIXEL_PADDING)
        x0 = max(0, x - SPILL_SEARCH_PIXEL_PADDING // 2)
        x1 = min(w, x + bw + SPILL_SEARCH_PIXEL_PADDING // 2)
        if y1 - y0 < 5 or x1 - x0 < 5:
            return None

        roi_rgb = rgb[y0:y1, x0:x1]
        roi_depth = depth[y0:y1, x0:x1]
        roi_hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_BGR2HSV)
        
        # Look ONLY for spills that match the barrel's colour to avoid mistaking floor lines
        sat_mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        for h_lo, s_lo, v_lo, h_hi, s_hi, v_hi in COLOUR_RANGES[barrel_colour]:
            if barrel_colour == 'black':
                # Black spills are extremely dark patches on the floor
                sat_mask |= cv2.inRange(roi_hsv, np.array([0, 0, 0]), np.array([180, 255, 50]))
            else:
                # For colored spills, we match the parent color but allow slightly lower saturation/value
                sat_mask |= cv2.inRange(roi_hsv, np.array([h_lo, 60, 40]), np.array([h_hi, 255, 255]))

        # Exclude depth pixels that are too far / NaN.
        finite = np.isfinite(roi_depth) & (roi_depth > cfg.depth_min) & (roi_depth < cfg.depth_max)
        sat_mask[~finite] = 0
        if not sat_mask.any():
            return None
        sat_mask = cv2.morphologyEx(sat_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        contours, _ = cv2.findContours(sat_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        K = self._intrinsics[cam_key]
        for cnt in contours:
            if cv2.contourArea(cnt) < 60:
                continue
            patch = np.zeros_like(sat_mask)
            cv2.drawContours(patch, [cnt], -1, 255, thickness=cv2.FILLED)
            patch_bool = patch.astype(bool) & finite
            if int(patch_bool.sum()) < 40:
                continue
            ys, xs = np.where(patch_bool)
            zs = roi_depth[ys, xs]
            # Convert ROI coords back to full-image coords for intrinsics.
            xs_full = xs + x0
            ys_full = ys + y0
            X = (xs_full - K['cx']) * zs / K['fx']
            Y = (ys_full - K['cy']) * zs / K['fy']
            pts_opt = np.stack([X, Y, zs], axis=1)
            pts_body = np.stack([pts_opt[:, 2], -pts_opt[:, 0], -pts_opt[:, 1]], axis=1)
            pts_map = self._transform_points(pts_body, frame_id)
            if pts_map is None:
                continue
            # Must lie on the floor.
            zs_map = pts_map[:, 2]
            on_floor = np.abs(zs_map - np.median(zs_map)) < SPILL_FLOOR_DEPTH_VAR * 4
            if on_floor.sum() < 40:
                continue
            floor_pts = pts_map[on_floor]
            if abs(float(np.median(floor_pts[:, 2]))) > SPILL_FLOOR_Z_MAX:
                continue
            # 3D footprint area: convex hull over XY of floor pts.
            xy = floor_pts[:, :2].astype(np.float32)
            if xy.shape[0] < 3:
                continue
            hull = cv2.convexHull(xy.reshape(-1, 1, 2))
            area = float(cv2.contourArea(hull))
            if area < SPILL_MIN_AREA_M2:
                continue
            centroid = floor_pts.mean(axis=0)
            if np.linalg.norm(centroid[:2] - barrel_anchor[:2]) > SPILL_MAX_DIST_TO_BARREL:
                continue
            return centroid
        return None

    # ----------------------------------------------------------- TF transform

    def _transform_points(self, pts_body: np.ndarray,
                          frame_id: str) -> Optional[np.ndarray]:
        """Apply a single TF lookup (frame_id → map) to a Nx3 array."""
        try:
            tf = self.tf_buffer.lookup_transform(
                'map', frame_id, RclTime(),
                timeout=Duration(seconds=0.1),
            )
        except Exception as e:
            # TF can be transiently unavailable during startup; don't spam.
            now = self.get_clock().now().nanoseconds
            if now - getattr(self, '_last_tf_warn_ns', 0) > 2_000_000_000:
                self._last_tf_warn_ns = now
                self.get_logger().warn(f'TF lookup failed for {frame_id} → map: {e}')
            return None
        q = tf.transform.rotation
        t = tf.transform.translation
        R = quat_to_mat(q.x, q.y, q.z, q.w)
        return (pts_body @ R.T) + np.array([t.x, t.y, t.z])

    # ----------------------------------------------------------------- output

    def _show_debug(self, cam_key: str, rgb: np.ndarray, candidates: list) -> None:
        if not hasattr(self, '_vis_history'):
            self._vis_history = {'oakd': {}, 'top': {}}

        # Decrement decay counter for existing items
        history = self._vis_history[cam_key]
        for colour in list(history.keys()):
            bbox, frames_left = history[colour]
            if frames_left <= 1:
                del history[colour]
            else:
                history[colour] = (bbox, frames_left - 1)

        # Update history with current frame's detections
        for cand in candidates:
            history[cand['colour']] = (cand['bbox'], 8)  # persist for 8 frames

        # Draw all active visual detections in history
        vis = rgb.copy()
        for colour, (bbox, _) in history.items():
            x, y, w_, h_ = bbox
            r, g, b = COLOUR_RGB.get(colour, (0.8, 0.8, 0.8))
            colour_bgr = (int(b * 255), int(g * 255), int(r * 255))
            cv2.rectangle(vis, (x, y), (x + w_, y + h_), colour_bgr, 2)
            cv2.putText(vis, colour, (x, max(15, y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour_bgr, 1)

        wname = 'barrels_oakd' if cam_key == 'oakd' else 'barrels_top'
        self._last_vis[cam_key] = vis
        cv2.imshow(wname, vis)

    def _publish(self) -> None:
        confirmed = self.barrel_map.confirmed_landmarks()

        # Stable hash of (id, colour, orientation, leaking) — only republish on change.
        sig_payload = tuple(
            (lm.id, lm.colour, lm.orientation, lm.leaking,
             round(float(lm.position[0]), 2), round(float(lm.position[1]), 2))
            for lm in confirmed
        )
        sig = hash(sig_payload)
        if sig == self._last_publish_hash:
            return
        self._last_publish_hash = sig

        now = self.get_clock().now().to_msg()
        ma = MarkerArray()
        for lm in confirmed:
            r, g, b = COLOUR_RGB.get(lm.colour, COLOUR_RGB['unknown'])

            # Cylinder.
            cyl = Marker()
            cyl.header.frame_id = 'map'
            cyl.header.stamp = now
            cyl.ns = NS_BARREL
            cyl.id = lm.id
            cyl.type = Marker.CYLINDER
            cyl.action = Marker.ADD
            cyl.pose.position.x = float(lm.position[0])
            cyl.pose.position.y = float(lm.position[1])
            cyl.pose.position.z = float(lm.position[2])
            if lm.orientation == 'horizontal':
                # Lay the cylinder on its side; orient along world X for now.
                # (We have no reliable per-barrel yaw measurement; the centroid
                # placement is what matters for the report.)
                cyl.pose.orientation = axis_to_quat(np.array([1.0, 0.0, 0.0]))
            else:
                cyl.pose.orientation.w = 1.0
            cyl.scale = Vector3(x=0.55, y=0.55, z=0.88)
            cyl.color = ColorRGBA(r=r, g=g, b=b, a=0.85)
            cyl.lifetime.sec = 0
            ma.markers.append(cyl)

            # Label.
            lbl = Marker()
            lbl.header.frame_id = 'map'
            lbl.header.stamp = now
            lbl.ns = NS_LABEL
            lbl.id = lm.id
            lbl.type = Marker.TEXT_VIEW_FACING
            lbl.action = Marker.ADD
            lbl.pose.position.x = float(lm.position[0])
            lbl.pose.position.y = float(lm.position[1])
            lbl.pose.position.z = float(lm.position[2]) + 0.7
            lbl.pose.orientation.w = 1.0
            lbl.scale = Vector3(x=0.0, y=0.0, z=0.18)
            lbl.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            leak_suffix = ' / LEAKING' if lm.leaking else ''
            lbl.text = f'Barrel {lm.id}: {lm.colour}/{lm.orientation}{leak_suffix}'
            lbl.lifetime.sec = 0
            ma.markers.append(lbl)

            # Spill marker (only if confirmed leaking and we have a position).
            if lm.leaking and lm.spill_position is not None:
                sp = Marker()
                sp.header.frame_id = 'map'
                sp.header.stamp = now
                sp.ns = NS_SPILL
                sp.id = lm.id
                sp.type = Marker.SPHERE
                sp.action = Marker.ADD
                sp.pose.position.x = float(lm.spill_position[0])
                sp.pose.position.y = float(lm.spill_position[1])
                sp.pose.position.z = float(lm.spill_position[2])
                sp.pose.orientation.w = 1.0
                sp.scale = Vector3(x=0.3, y=0.3, z=0.05)
                sp.color = ColorRGBA(r=1.0, g=0.1, b=0.1, a=0.9)
                sp.lifetime.sec = 0
                ma.markers.append(sp)

        if ma.markers:
            self.markers_pub.publish(ma)

        # JSON inspection report.
        payload = []
        for lm in confirmed:
            entry = {
                'id': lm.id,
                'colour': lm.colour,
                'orientation': lm.orientation,
                'leaking': bool(lm.leaking),
                'position': [float(lm.position[0]), float(lm.position[1]), float(lm.position[2])],
            }
            if lm.leaking and lm.spill_position is not None:
                entry['spill_position'] = [
                    float(lm.spill_position[0]),
                    float(lm.spill_position[1]),
                    float(lm.spill_position[2]),
                ]
            payload.append(entry)
        self.inspections_pub.publish(String(data=json.dumps(payload)))


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def main() -> None:
    rclpy.init(args=None)
    node = BarrelDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
