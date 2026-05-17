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

from scipy.spatial import KDTree
from scipy.interpolate import splprep, splev

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
        # Convert ROS Image messages to OpenCV images
        rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
        depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='32FC1')
        
        # Make explicit copies to avoid modifying originals
        rgb_display = rgb_image.copy()
        depth_for_processing = depth_image.copy()

        if self.received_camera_info:
            mask = ground_mask(depth_for_processing, self.fx, self.fy, self.cx_principal, self.cy_principal, self.ground_distance_threshold, self.ground_min_normal_y, self.ground_min_y_over_z)
            # ensure mask is single channel uint8 with values 0 or 255
            if mask is None:
                cv2.imshow('Lines',   rgb_display)
            else:
                m = mask.copy()
                if m.ndim == 3:
                    m = m[:, :, 0]
                m = (m > 0).astype(np.uint8)

                mask3 = np.repeat(m[:, :, None], 3, axis=2).astype(bool)

                
                # Detect and display lines on ground
                line_labels = line_mask(depth_for_processing, rgb_display, mask)
                if line_labels is not None and line_labels.max() > 0:
                    line_overlay = rgb_display.clip(0, 255).astype(np.uint8)
                    line_overlay[m > 0] = 0
                    
                    # Color map: class -> BGR color
                    color_map = {
                        1: (0, 200, 200),    # yellow in BGR
                        2: (0, 0, 200),      # red in BGR
                        3: (200, 0, 0),      # blue in BGR
                        4: (0, 200, 0),      # green in BGR
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

        # Extract transform as matrix — do this once, not per point
        t = transform.transform.translation
        q = transform.transform.rotation
        T = self._transform_to_matrix(t, q)

        colors_bgr = {
            1: (0, 200, 200),
            2: (0, 0, 200),
            3: (200, 0, 0),
            4: (0, 200, 0),
        }

        all_xyz = []
        all_rgb = []

        for class_id in [c for c in np.unique(line_labels) if c > 0]:
            ys, xs = np.where(line_labels == class_id)
            if len(xs) == 0:
                continue

            # Vectorized depth lookup — median over patch per pixel
            depths = np.array([
                self._sample_depth(depth_image, int(u), int(v))
                for u, v in zip(xs, ys)
            ])

            valid = depths > 0
            xs_v, ys_v, ds_v = xs[valid], ys[valid], depths[valid]
            if len(ds_v) == 0:
                continue

            # Backproject all at once
            cam_x = (xs_v - self.cx_principal) * ds_v / self.fx
            cam_y = (ys_v - self.cy_principal) * ds_v / self.fy
            cam_z = ds_v
        

            cam_x_ros = cam_z        # forward
            cam_y_ros = -cam_x       # left
            cam_z_ros = -cam_y       # up

            pts_cam = np.stack([cam_x_ros, cam_y_ros, cam_z_ros, np.ones_like(cam_z)], axis=1)
            pts_world = (T @ pts_cam.T).T

            all_xyz.append(pts_world[:, :3])

            b, g, r = colors_bgr.get(class_id, (255, 255, 255))
            all_rgb.append(np.tile([r, g, b], (len(pts_world), 1)))

        if not all_xyz:
            return

        xyz = np.concatenate(all_xyz, axis=0).astype(np.float32)
        rgb = np.concatenate(all_rgb, axis=0).astype(np.uint8)
        self.pc_pub.publish(self._make_pointcloud2(xyz, rgb, stamp))


    def _sample_depth(self, depth_image, u, v, patch=2):
        h, w = depth_image.shape[:2]
        region = depth_image[max(0,v-patch):min(h,v+patch+1),
                            max(0,u-patch):min(w,u+patch+1)].astype(np.float32)
        valid = region[np.isfinite(region) & (region > 0.1) & (region < 10.0)]
        return float(np.median(valid)) if len(valid) > 0 else 0.0


    def _transform_to_matrix(self, t, q):
        """Quaternion + translation → 4x4 transform matrix."""
        x, y, z, w = q.x, q.y, q.z, q.w
        T = np.array([
            [1-2*(y*y+z*z),   2*(x*y-z*w),   2*(x*z+y*w), t.x],
            [  2*(x*y+z*w), 1-2*(x*x+z*z),   2*(y*z-x*w), t.y],
            [  2*(x*z-y*w),   2*(y*z+x*w), 1-2*(x*x+y*y), t.z],
            [0, 0, 0, 1]
        ], dtype=np.float64)
        return T


    def _make_pointcloud2(self, xyz, rgb, stamp):
        """xyz: (N,3) float32, rgb: (N,3) uint8"""
        fields = [
            PointField(name='x',   offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y',   offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z',   offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        rgb_packed = (rgb[:, 0].astype(np.uint32) << 16 |
                    rgb[:, 1].astype(np.uint32) << 8  |
                    rgb[:, 2].astype(np.uint32))

        # Interleave xyz + rgb into packed float32 buffer
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
