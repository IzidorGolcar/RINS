#!/usr/bin/python3
"""
Geometry-first ring detector (v3).

Rings are a geometry problem, not a color-clustering problem. This node:
  1. Builds Canny edges from a CLAHE + bilateral-filtered grayscale image
     (auto-threshold based on median luminance → robust to lighting).
  2. Extracts hierarchical contours and fits an ellipse to each.
  3. Pairs outer+inner ellipses with matching centres — the strongest
     signal that a contour pair represents a real ring (a hole with a band).
  4. Validates each candidate with depth (ring-band std), interior hollowness,
     physical diameter, and camera-relative height.
  5. Samples ring color from the ring band *after* detection, so color is
     only a label, not a detection criterion.
"""

import os
os.environ.setdefault('QT_LOGGING_RULES', 'default.warning=false;qt.qpa.*=false')

import cv2
import math
import numpy as np
import rclpy
import rclpy.duration
import tf2_ros
import tf2_geometry_msgs  # noqa: F401  — registers PointStamped transform
import message_filters

from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import (
    QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile,
    QoSReliabilityPolicy, qos_profile_sensor_data,
)

from sensor_msgs.msg import CompressedImage, CameraInfo
from geometry_msgs.msg import PointStamped, Vector3
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from ring_map import RingMap


marker_qos = QoSProfile(
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)


class RingDetector(Node):
    TARGET_FRAME = 'map'

    # --- real-time controls ---
    FRAME_SKIP = 2          # process every Nth frame
    PROC_WIDTH = 320        # downsample for detection (px wide)

    # --- edge detection ---
    CANNY_SIGMA = 0.33      # auto-threshold band around median
    CLAHE_CLIP = 2.0
    BILATERAL_D = 5
    MORPH_CLOSE_ITERS = 1

    # --- shape gates ---
    MIN_CONTOUR_POINTS = 12
    MIN_CONTOUR_AREA = 40
    MIN_ASPECT = 0.45
    MIN_CIRC = 0.55
    MAX_CIRC = 1.30
    BORDER_PX = 4

    # --- inner/outer contour pairing ---
    PAIR_CENTER_FRAC = 0.25  # centre gap < this × outer minor axis
    PAIR_SIZE_FRAC = 0.80    # inner major < this × outer major

    # --- depth + hollow ---
    MIN_DEPTH_SAMPLES = 10
    MAX_DEPTH_STD = 0.35
    HOLLOW_GAP = 0.06
    HOLLOW_RATIO = 0.30

    # --- physical plausibility ---
    MIN_DIAMETER = 0.05
    MAX_DIAMETER = 0.45
    MIN_DEPTH = 0.3
    MAX_DEPTH = 3.0
    H_CAM = 0.22            # TurtleBot4 Gemini height above floor
    MIN_HEIGHT = -0.3
    MAX_HEIGHT = 1.85

    def __init__(self):
        super().__init__('ring_detector_v3')

        self._frame_count = 0

        self.declare_parameters('', [
            ('rgb_topic',    '/gemini/color/image_raw/compressed'),
            ('depth_topic',  '/gemini/depth/image_raw/compressedDepth'),
            ('camera_frame', 'gemini_color_frame'),
        ])
        self._rgb_topic    = self.get_parameter('rgb_topic').value
        self._depth_topic  = self.get_parameter('depth_topic').value
        self._camera_frame = self.get_parameter('camera_frame').value

        self.rgb_sub = message_filters.Subscriber(
            self, CompressedImage, self._rgb_topic,
            qos_profile=qos_profile_sensor_data)
        self.depth_sub = message_filters.Subscriber(
            self, CompressedImage, self._depth_topic,
            qos_profile=qos_profile_sensor_data)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub], queue_size=5, slop=0.2)
        self.ts.registerCallback(self.stream_callback)

        self.fx = self.fy = None
        self.cx_p = self.cy_p = None
        cam_info_base = self._rgb_topic.replace('/compressed', '').rsplit('/', 1)[0]
        cam_info_topic = cam_info_base + '/camera_info'
        self.create_subscription(CameraInfo, cam_info_topic,
                                 self.cam_info_callback, 10)
        self.create_subscription(CameraInfo, cam_info_topic,
                                 self.cam_info_callback, qos_profile_sensor_data)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.ring_pub = self.create_publisher(MarkerArray, '/ring_markers', marker_qos)
        self.ring_map = RingMap()

        self.clahe = cv2.createCLAHE(clipLimit=self.CLAHE_CLIP, tileGridSize=(8, 8))

        cv2.namedWindow('Detections', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Edges', cv2.WINDOW_NORMAL)

        self.get_logger().info(
            'Ring detector v3 (geometry-first) initialised. '
            f'rgb={self._rgb_topic}, depth={self._depth_topic}, '
            f'frame={self._camera_frame}')

    # ---------------- CAMERA INFO ----------------
    def cam_info_callback(self, msg: CameraInfo):
        if self.fx is None:
            self.fx = msg.k[0]
            self.fy = msg.k[4]
            self.cx_p = msg.k[2]
            self.cy_p = msg.k[5]
            self.get_logger().info(
                f'Intrinsics: fx={self.fx:.1f} fy={self.fy:.1f} '
                f'cx={self.cx_p:.1f} cy={self.cy_p:.1f}')

    # ---------------- MAIN CALLBACK ----------------
    def stream_callback(self, rgb_msg: CompressedImage, depth_msg: CompressedImage):
        self._frame_count += 1
        if self._frame_count % self.FRAME_SKIP != 0:
            return

        # Decode JPEG/PNG directly; avoids cv_bridge's cvtColor2 quirks on Jazzy.
        try:
            rgb_payload = np.frombuffer(rgb_msg.data, np.uint8)
            cv_image = cv2.imdecode(rgb_payload, cv2.IMREAD_COLOR)
            depth_payload = np.frombuffer(bytes(depth_msg.data)[12:], np.uint8)
            raw_depth = cv2.imdecode(depth_payload, cv2.IMREAD_UNCHANGED)
        except Exception as e:
            self.get_logger().error(f'Frame decode failed: {e}')
            return

        if cv_image is None or raw_depth is None:
            return
        if cv_image.size == 0 or raw_depth.size == 0:
            return
        if cv_image.ndim != 3 or cv_image.shape[2] != 3:
            self.get_logger().warn(f'Unexpected RGB shape {cv_image.shape}')
            return

        depth_image = (raw_depth.astype(np.float32) / 1000.0
                       if raw_depth.dtype == np.uint16
                       else raw_depth.astype(np.float32))

        if self.fx is None:
            cv2.imshow('Detections', cv_image)
            cv2.waitKey(1)
            return

        if depth_image.shape[:2] != cv_image.shape[:2]:
            depth_image = cv2.resize(
                depth_image,
                (cv_image.shape[1], cv_image.shape[0]),
                interpolation=cv2.INTER_NEAREST)

        self.detect_rings(cv_image, depth_image, rgb_msg.header.stamp)
        cv2.waitKey(1)

    # ---------------- DETECTION CORE ----------------
    def detect_rings(self, img_rgb, img_depth, stamp):
        H0, W0 = img_rgb.shape[:2]
        if W0 > self.PROC_WIDTH:
            scale = self.PROC_WIDTH / float(W0)
            small = cv2.resize(img_rgb, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_AREA)
        else:
            scale = 1.0
            small = img_rgb
        Hs, Ws = small.shape[:2]
        inv_scale = 1.0 / scale

        # CLAHE on luminance (lighting robustness) + bilateral (kills texture,
        # preserves edges) are the two big wins over a plain Gaussian blur.
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = self.clahe.apply(gray)
        smooth = cv2.bilateralFilter(gray, d=self.BILATERAL_D,
                                     sigmaColor=30, sigmaSpace=30)

        # Canny thresholds keyed off the frame's median luminance → adaptive.
        med = float(np.median(smooth))
        lo = int(max(0.0, (1.0 - self.CANNY_SIGMA) * med))
        hi = int(min(255.0, (1.0 + self.CANNY_SIGMA) * med))
        edges = cv2.Canny(smooth, lo, hi)

        # One step of morphological closing to re-join small edge gaps.
        k3 = np.ones((3, 3), np.uint8)
        edges_c = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k3,
                                   iterations=self.MORPH_CLOSE_ITERS)
        cv2.imshow('Edges', edges_c)

        # RETR_CCOMP gives a 2-level hierarchy: outer contours + their holes.
        contours, hier = cv2.findContours(
            edges_c, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        if not contours or hier is None:
            self.display_detections(img_rgb, [])
            return

        # Fit an ellipse to every contour that passes quick shape gates.
        candidates = []  # list of (idx, ellipse, cnt_area)
        for i, cnt in enumerate(contours):
            if len(cnt) < self.MIN_CONTOUR_POINTS:
                continue
            cnt_area = cv2.contourArea(cnt)
            if cnt_area < self.MIN_CONTOUR_AREA:
                continue
            try:
                e = cv2.fitEllipse(cnt)
            except cv2.error:
                continue
            (cx, cy), (a1, a2), _ = e
            cxi, cyi = int(cx), int(cy)
            if (cxi < self.BORDER_PX or cyi < self.BORDER_PX
                    or cxi >= Ws - self.BORDER_PX
                    or cyi >= Hs - self.BORDER_PX):
                continue
            major = max(a1, a2)
            minor = min(a1, a2)
            if major <= 0 or minor / major < self.MIN_ASPECT:
                continue
            ellipse_area = math.pi * 0.25 * major * minor
            if ellipse_area <= 0:
                continue
            circ = cnt_area / ellipse_area
            if not (self.MIN_CIRC < circ < self.MAX_CIRC):
                continue
            candidates.append((i, e, cnt_area))

        # Pair outer+inner ellipses: real rings produce TWO concentric contours
        # (one on the outer edge, one on the inner edge). Singletons can still
        # pass later if depth confirms hollow interior — so pairing is a boost,
        # not a hard requirement.
        paired = set()
        for (ia, ea, _) in candidates:
            (cax, cay), (a1a, a2a), _ = ea
            a_major = max(a1a, a2a)
            a_minor = min(a1a, a2a)
            for (ib, eb, _) in candidates:
                if ia == ib:
                    continue
                (cbx, cby), (a1b, a2b), _ = eb
                b_major = max(a1b, a2b)
                if b_major >= a_major * self.PAIR_SIZE_FRAC:
                    continue
                if math.hypot(cax - cbx, cay - cby) > a_minor * self.PAIR_CENTER_FRAC:
                    continue
                paired.add(ia)
                paired.add(ib)
                break

        # Process paired candidates first (higher-confidence), then the rest.
        ordered = sorted(candidates,
                         key=lambda c: (c[0] not in paired, -c[2]))

        counts = {'cand': len(candidates), 'depth': 0, 'hollow': 0,
                  'size': 0, 'height': 0, 'ok': 0}
        rings = []
        seen_centres = []

        for (i, e, _) in ordered:
            (cx_s, cy_s), (a1, a2), ang = e
            cx_f = cx_s * inv_scale
            cy_f = cy_s * inv_scale
            major_f = max(a1, a2) * inv_scale
            minor_f = min(a1, a2) * inv_scale

            # Merge duplicates (outer + inner contour of same ring).
            if any(math.hypot(cx_f - sx, cy_f - sy) < 0.5 * minor_f
                   for sx, sy in seen_centres):
                continue

            band_mask = self._ring_band_mask(
                img_depth.shape[:2], (cx_f, cy_f), (major_f, minor_f), ang)
            band_depths = img_depth[band_mask > 0]
            valid = band_depths[np.isfinite(band_depths) & (band_depths > 0.1)]
            if valid.size < self.MIN_DEPTH_SAMPLES:
                continue
            if float(np.std(valid)) > self.MAX_DEPTH_STD:
                continue
            z = float(np.median(valid))
            if not (self.MIN_DEPTH <= z <= self.MAX_DEPTH):
                continue
            counts['depth'] += 1

            if not self._is_hollow(img_depth, int(cx_f), int(cy_f),
                                   (major_f, minor_f), z):
                # Paired contours have already proven hollowness geometrically.
                if i not in paired:
                    continue
            counts['hollow'] += 1

            phys_d = (major_f * z) / self.fx
            if not (self.MIN_DIAMETER < phys_d < self.MAX_DIAMETER):
                continue
            counts['size'] += 1

            h_est = self.H_CAM - ((cy_f - self.cy_p) * z) / self.fy
            if not (self.MIN_HEIGHT < h_est < self.MAX_HEIGHT):
                continue
            counts['height'] += 1
            counts['ok'] += 1

            bgr = self._sample_band_color(img_rgb, band_mask)
            full_ellipse = ((cx_f, cy_f), (major_f, minor_f), ang)
            rings.append({
                'ellipse': full_ellipse,
                'color': bgr,
                'depth': z,
                'paired': i in paired,
            })
            seen_centres.append((cx_f, cy_f))

        if counts['cand']:
            self.get_logger().info(
                f"rings cand={counts['cand']} depth={counts['depth']} "
                f"hollow={counts['hollow']} size={counts['size']} "
                f"height={counts['height']} ok={counts['ok']}",
                throttle_duration_sec=2.0)

        self.display_detections(img_rgb, rings)
        self.localize(rings, stamp)

    # ---------------- MASKS / SAMPLING ----------------
    def _ring_band_mask(self, shape, centre, axes, angle):
        """Full-resolution filled annulus = outer ellipse − inner ellipse."""
        H, W = shape
        mask = np.zeros((H, W), np.uint8)
        cx, cy = int(centre[0]), int(centre[1])
        a_out = (int(axes[0] / 2), int(axes[1] / 2))
        # Inner boundary at ~64% of outer radius so the band is a clear strip.
        a_in = (max(1, int(axes[0] * 0.32)), max(1, int(axes[1] * 0.32)))
        cv2.ellipse(mask, (cx, cy), a_out, angle, 0, 360, 255, -1)
        cv2.ellipse(mask, (cx, cy), a_in,  angle, 0, 360, 0,   -1)
        return mask

    def _is_hollow(self, img_depth, cx, cy, axes, ring_depth):
        """Interior is hollow if it's mostly farther than the band or invalid."""
        H, W = img_depth.shape[:2]
        inner_r = max(2, int(min(axes) * 0.25))
        y0, y1 = max(0, cy - inner_r), min(H, cy + inner_r + 1)
        x0, x1 = max(0, cx - inner_r), min(W, cx + inner_r + 1)
        patch = img_depth[y0:y1, x0:x1]
        if patch.size == 0:
            return False
        invalid = np.sum(~np.isfinite(patch) | (patch <= 0.05))
        farther = np.sum(np.isfinite(patch) & (patch > ring_depth + self.HOLLOW_GAP))
        return (invalid + farther) / float(patch.size) >= self.HOLLOW_RATIO

    def _sample_band_color(self, img_rgb, band_mask):
        pixels = img_rgb[band_mask > 0]
        if pixels.size == 0:
            return (0, 0, 0)
        med = np.median(pixels, axis=0)
        return tuple(int(v) for v in med)

    # ---------------- DISPLAY ----------------
    def display_detections(self, img_rgb, rings):
        out = img_rgb.copy()
        for r in rings:
            e = r['ellipse']
            col = r['color']
            cv2.ellipse(out, e, col, 2)
            cx, cy = int(e[0][0]), int(e[0][1])
            tag = 'RING*' if r.get('paired') else 'ring'
            cv2.putText(out, f'{tag} {r["depth"]:.2f}m',
                        (cx + 8, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
        cv2.putText(out,
            f'confirmed: {len(self.ring_map.confirmed_landmarks())}',
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow('Detections', out)

    # ---------------- LOCALIZATION ----------------
    def localize(self, rings, stamp):
        if self.fx is None or not rings:
            return
        stamp_time = Time.from_msg(stamp)
        use_stamp = self.tf_buffer.can_transform(
            self.TARGET_FRAME, self._camera_frame, stamp_time,
            timeout=rclpy.duration.Duration(seconds=0.05))
        header_stamp = stamp if use_stamp else Time().to_msg()

        for r in rings:
            (cx_px, cy_px), _, _ = r['ellipse']
            z = r['depth']
            X_cam = (cx_px - self.cx_p) * z / self.fx
            Y_cam = (cy_px - self.cy_p) * z / self.fy

            pt = PointStamped()
            pt.header.frame_id = self._camera_frame
            pt.header.stamp = header_stamp
            pt.point.x = float(z)
            pt.point.y = float(-X_cam)
            pt.point.z = float(-Y_cam)
            try:
                pt_map = self.tf_buffer.transform(
                    pt, self.TARGET_FRAME,
                    timeout=rclpy.duration.Duration(seconds=0.1))
            except Exception as e:
                self.get_logger().warn(f'TF transform failed: {e}',
                                       throttle_duration_sec=5.0)
                continue
            pos = np.array([pt_map.point.x, pt_map.point.y, pt_map.point.z])
            self.ring_map.update(pos, r['color'])

        self._publish_confirmed()

    def _publish_confirmed(self):
        confirmed = self.ring_map.confirmed_landmarks()
        if not confirmed:
            return
        now = self.get_clock().now().to_msg()
        arr = MarkerArray()
        for lm in confirmed:
            b, g, r_ = [c / 255.0 for c in lm.color]
            s = Marker()
            s.header.frame_id = self.TARGET_FRAME
            s.header.stamp = now
            s.ns = 'confirmed_rings'
            s.id = lm.id
            s.type = Marker.SPHERE
            s.action = Marker.ADD
            s.pose.position.x = float(lm.position[0])
            s.pose.position.y = float(lm.position[1])
            s.pose.position.z = float(lm.position[2])
            s.pose.orientation.w = 1.0
            s.scale = Vector3(x=0.15, y=0.15, z=0.15)
            s.color = ColorRGBA(r=r_, g=g, b=b, a=1.0)
            arr.markers.append(s)

            t = Marker()
            t.header.frame_id = self.TARGET_FRAME
            t.header.stamp = now
            t.ns = 'confirmed_rings_labels'
            t.id = lm.id
            t.type = Marker.TEXT_VIEW_FACING
            t.action = Marker.ADD
            t.pose.position.x = float(lm.position[0])
            t.pose.position.y = float(lm.position[1])
            t.pose.position.z = float(lm.position[2]) + 0.25
            t.pose.orientation.w = 1.0
            t.scale = Vector3(x=0.0, y=0.0, z=0.15)
            t.color = ColorRGBA(r=r_, g=g, b=b, a=1.0)
            t.text = f'Ring {lm.id}'
            arr.markers.append(t)
        self.ring_pub.publish(arr)


def main():
    rclpy.init()
    node = RingDetector()
    try:
        rclpy.spin(node)
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
