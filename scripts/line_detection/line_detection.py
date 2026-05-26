#!/usr/bin/env python3

import os
import yaml
import math
import threading
from pathlib import Path
from datetime import datetime

import numpy as np
import cv2

import message_filters
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.qos import (QoSProfile, ReliabilityPolicy, HistoryPolicy,
                        QoSDurabilityPolicy, QoSReliabilityPolicy,
                        QoSHistoryPolicy)

import tf2_ros
import tf2_geometry_msgs  # registers PointStamped transform support

from cv_bridge import CvBridge

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped, PoseStamped, Point, Quaternion
from std_msgs.msg import String, Bool, Header
from visualization_msgs.msg import Marker, MarkerArray
from builtin_interfaces.msg import Duration as DurationMsg
from util import ground_mask, line_mask

from sensor_msgs.msg import PointCloud2, PointField
import struct

LINE_COLORS = {
    1: (1.0, 0.8, 0.0, 1.0),  # yellow
    2: (1.0, 0.0, 0.0, 1.0),  # red
    3: (0.0, 0.4, 1.0, 1.0),  # blue
    4: (0.0, 0.8, 0.0, 1.0),  # green
}

# Class IDs (mirror util.line_mask).
CLS_YELLOW = 1
CLS_RED    = 2
CLS_BLUE   = 3
CLS_GREEN  = 4

# Yellow-line safety zone in base_link frame (X forward, Y left).
# task2.py reverses on /lines/yellow_alert; the box has to be tight enough
# that the robot doesn't keep tripping the alert *while* reversing, but
# loose enough that we don't run wheels onto the line. At desired_linear_vel
# = 0.5 m/s with ~0.2 s detect→cancel latency, ~0.1 m of travel happens
# after the alert fires, so X_MAX ≥ 0.15 m is the practical floor.
YELLOW_DANGER_X_MAX  = 0.35  # m forward (was 0.6)
YELLOW_DANGER_HALF_Y = 0.25  # m to either side (was 0.4)
YELLOW_DANGER_MIN_POINTS = 8

# Blue-line follow geometry.
BLUE_LOOKAHEAD    = 0.7      # m ahead along the PCA-fitted line
BLUE_MIN_POINTS   = 15

# A red / green cell line counts as "this is the cell entrance" only when
# its centroid is within this radius of the robot. Stops a sliver of red
# seen at 4 m from accidentally locking the cell pose to a noisy spot.
CELL_PROXIMITY_R  = 1.5      # m, robot → cell centroid
CELL_MIN_POINTS   = 30


_LATCHED_QOS = QoSProfile(
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)


class LineDetector(Node):

    def __init__(self):
        super().__init__("line_detection")

        self.bridge = CvBridge()
        self.ground_distance_threshold = 0.0001
        self.ground_min_normal_y = 0.55
        self.ground_min_y_over_z = 1.1

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.pc_pub = self.create_publisher(PointCloud2, '/line_detector/points', 10)
        self.class_pubs = {
            1: self.create_publisher(PointCloud2, '/line_detector/yellow', 10),
            2: self.create_publisher(PointCloud2, '/line_detector/red',    10),
            3: self.create_publisher(PointCloud2, '/line_detector/blue',   10),
            4: self.create_publisher(PointCloud2, '/line_detector/green',  10),
        }

        self.rgb_sub = message_filters.Subscriber(self, Image, "/oakd/rgb/preview/image_raw")
        self.depth_sub = message_filters.Subscriber(self, Image, "/oakd/rgb/preview/depth")

        self.stream = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=1,
            slop=0.1
        )

        self._depth_med_cache = None
        self._last_tf = None
        self._last_tf_stamp = None


        self.stream.registerCallback(self.stream_callback)
        self.received_camera_info = False
        self.fx = self.fy = None
        self.cx_principal = self.cy_principal = None
        self.cam_info_sub = self.create_subscription(
            CameraInfo,
            '/oakd/rgb/preview/camera_info',
            self.cam_info_callback,
            QoSProfile(depth=1)
        )
        cv2.namedWindow('Lines', cv2.WINDOW_NORMAL)


    def cam_info_callback(self, msg):
        if self.fx is None:
            self.fx = msg.k[0]
            self.fy = msg.k[4]
            self.cx_principal = msg.k[2]
            self.cy_principal = msg.k[5]
            self.get_logger().info(
                f'Camera intrinsics loaded: fx={self.fx:.2f}, fy={self.fy:.2f}, '
                f'cx={self.cx_principal:.2f}, cy={self.cy_principal:.2f}'
            )
            self.received_camera_info = True


    def stream_callback(self, rgb_msg, depth_msg):
        rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
        depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='32FC1')

        rgb_display = rgb_image.copy()
        depth_for_processing = depth_image.copy()

        if self.received_camera_info:
            mask = ground_mask(
                depth_for_processing,
                self.fx, self.fy,
                self.cx_principal, self.cy_principal,
                self.ground_distance_threshold,
                self.ground_min_normal_y,
                self.ground_min_y_over_z
            )

            if mask is None:
                cv2.imshow('Lines', rgb_display)
            else:
                m = mask.copy()
                if m.ndim == 3:
                    m = m[:, :, 0]
                m = (m > 0).astype(np.uint8)

                line_labels = line_mask(depth_for_processing, rgb_display, mask)
                if line_labels is not None and line_labels.max() > 0:
                    line_overlay = rgb_display.clip(0, 255).astype(np.uint8)
                    line_overlay[m > 0] = 0

                    color_map = {
                        1: (0, 200, 200),   # yellow in BGR
                        2: (0, 0, 200),     # red in BGR
                        3: (200, 0, 0),     # blue in BGR
                        4: (0, 200, 0),     # green in BGR
                    }
                    for class_id, line_color in color_map.items():
                        class_mask = (line_labels == class_id).astype(np.uint8)
                        if class_mask.sum() > 0:
                            line_overlay[class_mask > 0] = line_color

                    cv2.imshow('Lines', line_overlay)
                    self._process_lines(line_labels, rgb_image, depth_for_processing)

                else:
                    line_overlay = rgb_display.clip(0, 255).astype(np.uint8)
                    line_overlay[m > 0] = 0
                    cv2.imshow('Lines', line_overlay)

        cv2.waitKey(1)


    # At the top of the class __init__, after existing setup:

    def _skeletonize(self, mask_255: np.ndarray) -> np.ndarray:
        """
        Replace the Zhang-Suen loop with cv2.ximgproc.thinning.
        Falls back to the morphological skeleton if ximgproc is absent.
        Same output: uint8, 0/255, 1-px-wide skeleton.
        """
        try:
            # cv2.ximgproc is in opencv-contrib-python. This is 10-50× faster
            # than the pure-NumPy Zhang-Suen loop and handles junctions/curves.
            import cv2.ximgproc as xip
            return xip.thinning(mask_255, thinningType=xip.THINNING_ZHANGSUEN)
        except (ImportError, AttributeError):
            # Fallback: morphological skeleton via erode+open, no iteration loop.
            # Faster than the manual Zhang-Suen even though it's slightly less
            # precise at junctions — good enough for 1-px-wide floor lines.
            img = mask_255.copy()
            skel = np.zeros_like(img)
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
            while True:
                eroded = cv2.erode(img, kernel)
                opened = cv2.dilate(eroded, kernel)
                skel |= cv2.subtract(img, opened)
                img = eroded
                if cv2.countNonZero(img) == 0:
                    break
            return skel


    def _process_lines(self, line_labels, rgb_image, depth_image):
        if not self.received_camera_info:
            return

        stamp = self.get_clock().now().to_msg()

        # ── 1. TF lookup with caching ─────────────────────────────────────────
        now_sec = stamp.sec + stamp.nanosec * 1e-9
        last_sec = (self._last_tf_stamp or 0)
        if self._last_tf is None or (now_sec - last_sec) > 0.05:
            try:
                tf = self.tf_buffer.lookup_transform(
                    'map', 'oakd_link',
                    Time(), timeout=Duration(seconds=0.05)
                )
                self._last_tf = self._transform_to_matrix(
                    tf.transform.translation, tf.transform.rotation
                )
                self._last_tf_stamp = now_sec
            except Exception as e:
                self.get_logger().warn(f'TF failed: {e}', throttle_duration_sec=2.0)
                if self._last_tf is None:
                    return   # no cached fallback yet

        T = self._last_tf

        # ── 2. Median blur ONCE for the whole frame ───────────────────────────
        depth_med = cv2.medianBlur(depth_image.astype(np.float32), 5)

        colors_bgr = {1: (0,200,200), 2: (0,0,200), 3: (200,0,0), 4: (0,200,0)}
        all_xyz, all_rgb = [], []

        # Pre-compute unique classes (skip background 0)
        unique_classes = [int(c) for c in np.unique(line_labels) if c > 0]

        for class_id in unique_classes:
            # ── 3. Mask + thin ────────────────────────────────────────────────
            class_mask = ((line_labels == class_id) * 255).astype(np.uint8)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            class_mask = cv2.erode(class_mask, kernel, iterations=1)
            thin = self._skeletonize(class_mask)

            margin = 4
            if thin.shape[0] > 2 * margin and thin.shape[1] > 2 * margin:
                thin[:margin, :] = 0; thin[-margin:, :] = 0
                thin[:, :margin] = 0; thin[:, -margin:]  = 0

            ys, xs = np.where(thin > 0)
            if len(xs) == 0:
                continue

            # ── 4. Depth + consistency (reuse pre-computed median) ────────────
            depths = depth_image[ys, xs].astype(np.float32)
            valid = np.isfinite(depths) & (depths > 0.1) & (depths < 10.0)
            if not valid.any():
                continue
            xs_v, ys_v, ds_v = xs[valid], ys[valid], depths[valid]

            med_vals = depth_med[ys_v, xs_v]
            tol = np.maximum(0.05, 0.05 * ds_v)
            keep = np.abs(ds_v - med_vals) <= tol
            if not keep.any():
                continue
            xs_v, ys_v, ds_v = xs_v[keep], ys_v[keep], ds_v[keep]

            # ── 5. Backproject — unchanged, already vectorised ─────────────────
            cam_x = (xs_v - self.cx_principal) * ds_v / self.fx
            cam_y = (ys_v - self.cy_principal) * ds_v / self.fy
            ros_x, ros_y, ros_z = cam_z = ds_v, -cam_x, -cam_y

            pts = np.stack([ros_x, ros_y, ros_z, np.ones_like(ros_x)], axis=1)
            pts_world = (T @ pts.T).T

            class_xyz = pts_world[:, :3].astype(np.float32)
            b, g, r = colors_bgr.get(class_id, (255, 255, 255))
            class_rgb = np.broadcast_to(
                np.array([[r, g, b]], dtype=np.uint8), (len(class_xyz), 3)
            ).copy()

            # ── 6. Only publish if there are subscribers ──────────────────────
            if class_id in self.class_pubs:
                pub = self.class_pubs[class_id]
                if pub.get_subscription_count() > 0:
                    pub.publish(self._make_pointcloud2(class_xyz, class_rgb, stamp))

            all_xyz.append(class_xyz)
            all_rgb.append(class_rgb)

        if all_xyz and self.pc_pub.get_subscription_count() > 0:
            xyz = np.concatenate(all_xyz, axis=0)
            rgb = np.concatenate(all_rgb, axis=0)
            self.pc_pub.publish(self._make_pointcloud2(xyz, rgb, stamp))

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _transform_to_matrix(self, t, q):
        """Quaternion + translation → 4×4 homogeneous transform matrix."""
        x, y, z, w = q.x, q.y, q.z, q.w
        return np.array([
            [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w), t.x],
            [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w), t.y],
            [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y), t.z],
            [0, 0, 0, 1],
        ], dtype=np.float64)


    def _make_pointcloud2(self, xyz, rgb, stamp):
        """
        xyz : (N,3) float32
        rgb : (N,3) uint8  — channel order R, G, B
        """
        fields = [
            PointField(name='x',   offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y',   offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z',   offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        rgb_packed = (rgb[:, 0].astype(np.uint32) << 16 |
                      rgb[:, 1].astype(np.uint32) << 8  |
                      rgb[:, 2].astype(np.uint32))

        buf = np.zeros((len(xyz), 4), dtype=np.float32)
        buf[:, :3] = xyz
        buf[:, 3].view(np.uint32)[:] = rgb_packed

        msg = PointCloud2()
        msg.header.stamp = stamp
        msg.header.frame_id = 'map'
        msg.height = 1
        msg.width = len(xyz)
        msg.fields = fields
        msg.is_bigendian = False
        msg.point_step = 16
        msg.row_step = 16 * len(xyz)
        msg.data = buf.tobytes()
        msg.is_dense = True
        return msg


def main(args=None):
    rclpy.init(args=None)
    rd_node = LineDetector()
    rclpy.spin(rd_node)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()