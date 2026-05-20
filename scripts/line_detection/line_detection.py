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
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

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


    # ── Zhang-Suen thinning (no skimage, no ximgproc) ───────────────────────

    def _skeletonize(self, mask_255):
        """
        Zhang-Suen thinning. Pure numpy/cv2.
        mask_255: uint8 binary image (0 / 255).
        Returns uint8 binary image with 1-px-wide skeleton (0 / 255).
        """
        img = (mask_255 // 255).astype(np.uint8)
        prev = np.zeros_like(img)

        def neighbors(x):
            return [x[:-2, 1:-1], x[:-2, 2:], x[1:-1, 2:], x[2:, 2:],
                    x[2:, 1:-1], x[2:, :-2], x[1:-1, :-2], x[:-2, :-2]]

        while True:
            P = [n.copy() for n in neighbors(img)]
            N = sum(P)
            ring = P + [P[0]]
            T = sum((ring[i] == 0) & (ring[i + 1] == 1) for i in range(8))

            interior = img[1:-1, 1:-1]
            cond_all = (interior == 1) & (N >= 2) & (N <= 6) & (T == 1)

            # Step 1
            step1 = cond_all & (P[0] * P[2] * P[4] == 0) & (P[2] * P[4] * P[6] == 0)
            marker = np.zeros_like(img)
            marker[1:-1, 1:-1] = step1
            img[marker == 1] = 0

            # Step 2
            P = [n.copy() for n in neighbors(img)]
            N = sum(P)
            ring = P + [P[0]]
            T = sum((ring[i] == 0) & (ring[i + 1] == 1) for i in range(8))
            interior = img[1:-1, 1:-1]
            cond_all = (interior == 1) & (N >= 2) & (N <= 6) & (T == 1)

            step2 = cond_all & (P[0] * P[2] * P[6] == 0) & (P[0] * P[4] * P[6] == 0)
            marker = np.zeros_like(img)
            marker[1:-1, 1:-1] = step2
            img[marker == 1] = 0

            if np.array_equal(img, prev):
                break
            prev = img.copy()

        return (img * 255).astype(np.uint8)


    # ── Main processing ──────────────────────────────────────────────────────

    def _process_lines(self, line_labels, rgb_image, depth_image):
        if not self.received_camera_info:
            return

        stamp = self.get_clock().now().to_msg()

        try:
            transform = self.tf_buffer.lookup_transform(
                'map', 'oakd_link',
                Time(), timeout=Duration(seconds=0.05)
            )
        except Exception as e:
            self.get_logger().warn(f'TF failed: {e}', throttle_duration_sec=2.0)
            return

        T = self._transform_to_matrix(
            transform.transform.translation,
            transform.transform.rotation
        )

        colors_bgr = {
            1: (0, 200, 200),   # yellow
            2: (0, 0, 200),     # red
            3: (200, 0, 0),     # blue
            4: (0, 200, 0),     # green
        }

        all_xyz = []
        all_rgb = []

        for class_id in [c for c in np.unique(line_labels) if c > 0]:
            class_mask = ((line_labels == class_id) * 255).astype(np.uint8)

            # Pre-erode slightly to help thinning converge faster on thick lines
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            class_mask = cv2.erode(class_mask, kernel, iterations=1)

            # Thin to 1-px skeleton — drastically reduces point count
            thin = self._skeletonize(class_mask)

            ys, xs = np.where(thin > 0)
            if len(xs) == 0:
                continue

            # Vectorized single-pixel depth lookup (skeleton centre is reliable)
            depths = depth_image[ys, xs].astype(np.float32)
            valid = np.isfinite(depths) & (depths > 0.1) & (depths < 10.0)
            xs_v = xs[valid]
            ys_v = ys[valid]
            ds_v = depths[valid]
            if len(ds_v) == 0:
                continue

            # Backproject to camera (optical) frame: Z forward, X right, Y down
            cam_x = (xs_v - self.cx_principal) * ds_v / self.fx
            cam_y = (ys_v - self.cy_principal) * ds_v / self.fy
            cam_z = ds_v

            # Optical → ROS body convention: X forward, Y left, Z up
            ros_x = cam_z
            ros_y = -cam_x
            ros_z = -cam_y

            # Transform all points to map frame in one matrix multiply
            pts = np.stack([ros_x, ros_y, ros_z, np.ones_like(ros_x)], axis=1)  # (N,4)
            pts_world = (T @ pts.T).T  # (N,4)

            class_xyz = pts_world[:, :3].astype(np.float32)
            b, g, r = colors_bgr.get(class_id, (255, 255, 255))
            class_rgb = np.tile([r, g, b], (len(class_xyz), 1)).astype(np.uint8)

            # Publish per-class topic
            if class_id in self.class_pubs:
                self.class_pubs[class_id].publish(
                    self._make_pointcloud2(class_xyz, class_rgb, stamp)
                )

            all_xyz.append(class_xyz)
            all_rgb.append(class_rgb)

        if not all_xyz:
            return

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