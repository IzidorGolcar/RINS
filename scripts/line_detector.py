#!/usr/bin/env python3
"""Floor-line perception for Task 2.

Subscribes to one downward camera (the arm-mounted ``top_camera``) and the
forward Oak-D, detects coloured floor lines by HSV thresholds, and publishes:

  /lines/yellow_alert     std_msgs/Bool         True while yellow is in the
                                                 imminent ROI (drives runtime
                                                 safety stop in task2.py)
  /lines/yellow_obstacles sensor_msgs/PointCloud2  yellow pixels projected to
                                                 base_link (flat-floor) — fed
                                                 into Nav2 as an obstacle
                                                 observation source.
  /lines/blue_target      geometry_msgs/PoseStamped  rolling target ~0.4 m
                                                 ahead along the blue line,
                                                 in base_link.
  /lines/cell_detected    std_msgs/String       'red' / 'green' when a thick
                                                 cell strip is straddled.
  /line_overlay           sensor_msgs/Image     debug overlay.

All thresholds are exposed as ROS 2 parameters so they can be tuned without
editing the file.
"""

from __future__ import annotations

import math
import struct

import cv2
import numpy as np
import rclpy
import tf2_ros
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Bool, Header, String


_COLOURS = {
    # H ranges in OpenCV (0–179).
    'yellow': [((20, 120, 120), (35, 255, 255))],
    'blue':   [((100, 120, 60), (130, 255, 255))],
    'red':    [((0, 120, 80), (10, 255, 255)),
               ((170, 120, 80), (180, 255, 255))],
    'green':  [((40, 100, 60), (85, 255, 255))],
}


def _make_pointcloud(points_xyz: np.ndarray, frame_id: str, stamp) -> PointCloud2:
    """Build a PointCloud2 from an (N, 3) float32 array."""
    msg = PointCloud2()
    msg.header = Header(stamp=stamp, frame_id=frame_id)
    msg.height = 1
    msg.width = points_xyz.shape[0]
    msg.fields = [
        PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = msg.point_step * msg.width
    msg.is_dense = True
    msg.data = points_xyz.astype(np.float32).tobytes()
    return msg


class LineDetector(Node):
    def __init__(self):
        super().__init__('line_detector')

        self.declare_parameters('', [
            ('down_camera_topic',  '/top_camera/rgb/preview/image_raw'),
            ('front_camera_topic', '/oakd/rgb/preview/image_raw'),
            ('camera_height',      0.35),   # m — top_camera height when arm parked
            ('camera_pitch',       1.5708), # rad — straight down (pi/2)
            ('camera_hfov',        1.25),   # rad — matches URDF's <horizontal_fov>
            # Imminent yellow ROI = a horizontal strip near the bottom of the down image.
            ('yellow_roi_y0', 0.55),
            ('yellow_roi_y1', 1.00),
            ('yellow_roi_min_pixels', 80),
            ('blue_lookahead', 0.4),         # m — how far ahead to project the blue target
            ('cell_min_pixels', 1500),       # min mask area to call a "cell straddle"
            ('publish_rate', 8.0),
        ])

        self.bridge = CvBridge()
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # State filled by callbacks
        self._latest_down: np.ndarray | None = None
        self._latest_front: np.ndarray | None = None
        self._latest_down_stamp = None

        # Publishers
        self.alert_pub  = self.create_publisher(Bool,        '/lines/yellow_alert',     10)
        self.cloud_pub  = self.create_publisher(PointCloud2, '/lines/yellow_obstacles', 10)
        self.blue_pub   = self.create_publisher(PoseStamped, '/lines/blue_target',      10)
        self.cell_pub   = self.create_publisher(String,      '/lines/cell_detected',    10)
        self.debug_pub  = self.create_publisher(Image,       '/line_overlay',           10)

        # Subscribers
        down_topic  = self.get_parameter('down_camera_topic').get_parameter_value().string_value
        front_topic = self.get_parameter('front_camera_topic').get_parameter_value().string_value
        self.create_subscription(Image, down_topic,  self._down_cb,  qos_profile_sensor_data)
        self.create_subscription(Image, front_topic, self._front_cb, qos_profile_sensor_data)

        rate = float(self.get_parameter('publish_rate').get_parameter_value().double_value)
        self.create_timer(1.0 / max(rate, 1.0), self._tick)

        self.get_logger().info(
            f'LineDetector up. down={down_topic} front={front_topic}')

    # -------------------------------------------------------------- callbacks

    def _down_cb(self, msg: Image) -> None:
        try:
            self._latest_down = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self._latest_down_stamp = msg.header.stamp
        except CvBridgeError as exc:
            self.get_logger().warn(f'down image conv failed: {exc}')

    def _front_cb(self, msg: Image) -> None:
        try:
            self._latest_front = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except CvBridgeError as exc:
            self.get_logger().warn(f'front image conv failed: {exc}')

    # ------------------------------------------------------------------- tick

    def _tick(self) -> None:
        if self._latest_down is None and self._latest_front is None:
            return

        masks_down = self._mask_all(self._latest_down) if self._latest_down is not None else {}
        masks_front = self._mask_all(self._latest_front) if self._latest_front is not None else {}

        alert = self._compute_yellow_alert(masks_down, masks_front)
        self.alert_pub.publish(Bool(data=alert))

        if 'yellow' in masks_down:
            cloud = self._project_to_base_link(masks_down['yellow'])
            if cloud is not None:
                self.cloud_pub.publish(cloud)

        if 'blue' in masks_down:
            self._publish_blue_target(masks_down['blue'])

        cell = self._detect_cell(masks_down)
        if cell is not None:
            self.cell_pub.publish(String(data=cell))

        if self._latest_down is not None:
            self._publish_overlay(self._latest_down, masks_down)

    # ----------------------------------------------------------------- masks

    def _mask_all(self, bgr: np.ndarray) -> dict[str, np.ndarray]:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        out: dict[str, np.ndarray] = {}
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        for name, ranges in _COLOURS.items():
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lo, hi in ranges:
                mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            out[name] = mask
        return out

    # ---------------------------------------------------------------- alerts

    def _compute_yellow_alert(self, down_masks: dict[str, np.ndarray],
                              front_masks: dict[str, np.ndarray]) -> bool:
        min_px = int(self.get_parameter('yellow_roi_min_pixels').get_parameter_value().integer_value)

        if 'yellow' in down_masks:
            m = down_masks['yellow']
            h = m.shape[0]
            y0 = int(self.get_parameter('yellow_roi_y0').get_parameter_value().double_value * h)
            y1 = int(self.get_parameter('yellow_roi_y1').get_parameter_value().double_value * h)
            if int(np.count_nonzero(m[y0:y1])) >= min_px:
                return True

        # Bottom 30% of the front camera = roughly 0.5–1.5 m ahead → early warning.
        if 'yellow' in front_masks:
            m = front_masks['yellow']
            h = m.shape[0]
            if int(np.count_nonzero(m[int(0.7 * h):])) >= min_px * 2:
                return True

        return False

    # ---------------------------------------------------- floor projection (yellow PointCloud)

    def _project_to_base_link(self, mask: np.ndarray) -> PointCloud2 | None:
        """Treat the down camera as a pinhole pointed straight down at fixed height,
        and convert mask pixels into points on the floor in base_link."""
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return None

        # Subsample to keep PointCloud size sane for nav2.
        if len(xs) > 800:
            sel = np.random.choice(len(xs), size=800, replace=False)
            ys, xs = ys[sel], xs[sel]

        h, w = mask.shape[:2]
        hfov = float(self.get_parameter('camera_hfov').get_parameter_value().double_value)
        cam_h = float(self.get_parameter('camera_height').get_parameter_value().double_value)

        # Pinhole, square pixels: focal in pixels.
        f = 0.5 * w / math.tan(0.5 * hfov)
        cx, cy = w / 2.0, h / 2.0

        # Camera optical frame: x right, y down, z forward.
        # Floor-projection assuming camera points straight down:
        #   X_floor (forward, +x base_link) = (cy - py) * cam_h / f
        #   Y_floor (left,    +y base_link) = (cx - px) * cam_h / f
        # That orientation matches the standard "rotated -pi/2 around Y" mount.
        px = xs.astype(np.float32)
        py = ys.astype(np.float32)
        x_base = (cy - py) * cam_h / f
        y_base = (cx - px) * cam_h / f
        z_base = np.zeros_like(x_base)

        pts = np.stack([x_base, y_base, z_base], axis=1)
        return _make_pointcloud(pts, 'base_link', self.get_clock().now().to_msg())

    # -------------------------------------------------- blue line target

    def _publish_blue_target(self, mask: np.ndarray) -> None:
        if int(np.count_nonzero(mask)) < 200:
            return
        ys, xs = np.where(mask > 0)
        # Fit line through the blue pixels (in image coords).
        pts = np.column_stack([xs, ys]).astype(np.float32)
        vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        # Walk along the line from the centroid towards the *upper* side
        # of the image (= away from the robot in the down view).
        if vy > 0:
            vx, vy = -vx, -vy

        h, w = mask.shape[:2]
        hfov = float(self.get_parameter('camera_hfov').get_parameter_value().double_value)
        cam_h = float(self.get_parameter('camera_height').get_parameter_value().double_value)
        lookahead = float(self.get_parameter('blue_lookahead').get_parameter_value().double_value)

        f = 0.5 * w / math.tan(0.5 * hfov)
        cx_p, cy_p = w / 2.0, h / 2.0

        # Step ~lookahead metres ahead in image space:
        # 1 m in front of robot ≈ f / cam_h pixels (along image -y).
        step_pixels = lookahead * f / cam_h
        ax = x0 + vx * step_pixels
        ay = y0 + vy * step_pixels

        x_base = (cy_p - ay) * cam_h / f
        y_base = (cx_p - ax) * cam_h / f

        pose = PoseStamped()
        pose.header.frame_id = 'base_link'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = float(x_base)
        pose.pose.position.y = float(y_base)
        # Yaw towards the target.
        yaw = math.atan2(y_base, x_base)
        pose.pose.orientation.z = math.sin(yaw / 2.0)
        pose.pose.orientation.w = math.cos(yaw / 2.0)
        self.blue_pub.publish(pose)

    # ----------------------------------------------------- working-cell detection

    def _detect_cell(self, masks: dict[str, np.ndarray]) -> str | None:
        min_px = int(self.get_parameter('cell_min_pixels').get_parameter_value().integer_value)
        for name in ('red', 'green'):
            if name in masks and int(np.count_nonzero(masks[name])) >= min_px:
                return name
        return None

    # ---------------------------------------------------------------- overlay

    def _publish_overlay(self, bgr: np.ndarray, masks: dict[str, np.ndarray]) -> None:
        vis = bgr.copy()
        tints = {
            'yellow': (0, 255, 255),
            'blue':   (255, 0,   0),
            'red':    (0, 0,   255),
            'green':  (0, 255, 0),
        }
        for name, m in masks.items():
            if name not in tints:
                continue
            colour = np.zeros_like(vis)
            colour[m > 0] = tints[name]
            vis = cv2.addWeighted(vis, 1.0, colour, 0.4, 0.0)
        try:
            self.debug_pub.publish(self.bridge.cv2_to_imgmsg(vis, 'bgr8'))
        except CvBridgeError as exc:
            self.get_logger().warn(f'overlay publish failed: {exc}')


def main():
    rclpy.init(args=None)
    node = LineDetector()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
