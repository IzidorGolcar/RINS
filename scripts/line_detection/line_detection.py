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


DEFAULT_HSV = {
    #         (H_low, S_low, V_low)   (H_high, S_high, V_high)
    "yellow": ((20,  100, 100),       (35,  255, 255)),
    "green":  ((40,   80,  80),       (80,  255, 255)),
    "red_lo": (( 0,  120, 120),       (10,  255, 255)),  # red wraps around 0°
    "red_hi": ((170, 120, 120),       (179, 255, 255)),  # upper red segment
}

class LineDetector(Node):

    def __init__(self):
        super().__init__("line_detection")

        self.bridge = CvBridge()
    
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
        cv2.namedWindow('Overlay', cv2.WINDOW_NORMAL)


    def stream_callback(self, rgb_msg, depth_msg):
        # Convert ROS Image messages to OpenCV images
        rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
        depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='32FC1')
        
        # Make explicit copies to avoid modifying originals
        rgb_display = rgb_image.copy()
        depth_for_processing = depth_image.copy()

        if not self.received_camera_info:
            # show blank until intrinsics are available
            cv2.imshow('Overlay', rgb_display)
        else:
            mask = self._ground_mask(depth_for_processing)
            # ensure mask is single channel uint8 with values 0 or 255
            if mask is None:
                cv2.imshow('Overlay', rgb_display)
                cv2.imshow('Lines',   rgb_display)
            else:
                m = mask.copy()
                if m.ndim == 3:
                    m = m[:, :, 0]
                m = (m > 0).astype(np.uint8)

                # create a colored mask (green)
                colored = np.zeros_like(rgb_display)
                colored[:, :] = (0, 255, 0)

                alpha = 0.4
                blended = cv2.addWeighted(rgb_display.astype(np.uint8), 1.0, colored, alpha, 0)

                # apply blended only where mask is set
                mask3 = np.repeat(m[:, :, None], 3, axis=2).astype(bool)
                overlay = rgb_display.copy()
                overlay[mask3] = blended[mask3]

                cv2.imshow('Overlay', overlay)
                
                # Detect and display lines on ground
                line_labels = self._line_mask(depth_for_processing, rgb_display, mask)
                if line_labels is not None and line_labels.max() > 0:
                    # Create line overlay with different colors per class
                    line_overlay = overlay.copy()
                    
                    # Color map: class -> BGR color
                    color_map = {
                        1: (255, 255, 0),    # cyan (yellow)
                        2: (0, 0, 255),      # red (red_lo)
                        3: (0, 0, 255),      # red (red_hi)
                        4: (0, 255, 0),      # green
                    }
                    
                    for class_id, line_color in color_map.items():
                        class_mask = (line_labels == class_id).astype(np.uint8)
                        if class_mask.sum() > 0:
                            line_overlay[class_mask > 0] = line_color
                    
                    cv2.imshow('Lines', line_overlay)
                else:
                    cv2.imshow('Lines', overlay)

        cv2.waitKey(1)


    def _line_mask(self, depth_image, rgb_image, ground_mask):
        """Segment colored lines (yellow, red, green) on the ground plane.
        Returns multiclass labels: 0=background, 1=yellow, 2=red_lo, 3=red_hi, 4=green.
        """
        if ground_mask is None:
            return None
        
        h, w = rgb_image.shape[:2]
        
        # Initialize multiclass label map (0 = background)
        line_labels = np.zeros((h, w), dtype=np.uint8)
        
        # Convert RGB to HSV for color-based segmentation
        hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
        
        # Restrict to ground region
        ground_binary = (ground_mask > 0).astype(np.uint8)
        
        # Create multiclass labels for each color
        color_class_map = {'yellow': 1, 'red_lo': 2, 'red_hi': 3, 'green': 4}
        
        for color_name, class_id in color_class_map.items():
            if color_name not in DEFAULT_HSV:
                continue
            hsv_lo, hsv_hi = DEFAULT_HSV[color_name]
            color_mask = cv2.inRange(hsv_image, np.array(hsv_lo), np.array(hsv_hi))
            # Apply to ground region only
            color_mask = cv2.bitwise_and(color_mask, ground_binary)
            # Assign class label where mask is positive
            line_labels[color_mask > 0] = class_id
        
        # Visualize raw multiclass labels (apply colormap)
        if line_labels.max() > 0:
            label_vis = cv2.applyColorMap((line_labels.astype(np.uint8) * 60).clip(0, 255), cv2.COLORMAP_JET)
            cv2.imshow('Lines Raw', label_vis)
        
        return line_labels


    def _ground_mask(self, depth_image):
        # Compute per-pixel 3D coordinates (camera frame) and fit a ground plane with RANSAC.
        # Returns a uint8 (0/255) mask where ground pixels are 255.
        if not self.received_camera_info or self.fx is None:
            h, w = depth_image.shape[:2]
            return np.zeros((h, w), dtype=np.uint8)

        depth = np.array(depth_image, dtype=np.float32)
        if depth.ndim == 3:
            depth = depth[:, :, 0]

        # Mask invalid depths
        valid_mask = np.isfinite(depth) & (depth > 0.05) & (depth < 10.0)
        if valid_mask.sum() < 50:
            return np.zeros_like(depth, dtype=np.uint8)

        h, w = depth.shape

        # Camera intrinsics
        fx = float(self.fx)
        fy = float(self.fy)
        cx = float(self.cx_principal)
        cy = float(self.cy_principal)

        # Create pixel coordinate grids
        u, v = np.meshgrid(np.arange(w), np.arange(h))

        Z = depth
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy

        # Use lower part of image to sample ground points (focus on plausible ground region)
        sample_mask = valid_mask.copy()
        sample_mask[: int(h * 0.4), :] = False

        pts_indices = np.where(sample_mask)
        pts_count = pts_indices[0].shape[0]
        if pts_count < 50:
            return np.zeros_like(depth, dtype=np.uint8)

        # Build Nx3 array of points (X,Y,Z)
        pts = np.stack((X[pts_indices], Y[pts_indices], Z[pts_indices]), axis=1)

        # Subsample for RANSAC if too many points
        max_samples = 4000
        if pts.shape[0] > max_samples:
            idx = np.random.choice(pts.shape[0], max_samples, replace=False)
            pts_sample = pts[idx]
        else:
            pts_sample = pts

        # RANSAC plane fitting
        best_inliers = None
        best_plane = None
        iterations = 300
        distance_threshold = 0.0001  # meters
        rng = np.random.default_rng()
        N = pts_sample.shape[0]
        if N < 3:
            return np.zeros_like(depth, dtype=np.uint8)

        for _ in range(iterations):
            # pick 3 random distinct indices
            i1, i2, i3 = rng.choice(N, size=3, replace=False)
            p1 = pts_sample[i1]
            p2 = pts_sample[i2]
            p3 = pts_sample[i3]
            # compute normal
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)
            if norm < 1e-6:
                continue
            normal = normal / norm
            d = -np.dot(normal, p1)

            # distances of all sampled pts to plane
            distances = np.abs(np.dot(pts_sample, normal) + d)
            inliers = distances <= distance_threshold
            inlier_count = int(inliers.sum())
            if best_inliers is None or inlier_count > best_inliers:
                best_inliers = inlier_count
                best_plane = (normal.copy(), float(d))

        if best_plane is None:
            return np.zeros_like(depth, dtype=np.uint8)

        normal, d = best_plane

        # compute distance for all valid pixels
        pts_all = np.stack((X[valid_mask], Y[valid_mask], Z[valid_mask]), axis=1)
        distances_all = np.abs(np.dot(pts_all, normal) + d)
        ground_mask_vals = np.zeros_like(depth, dtype=bool)
        ground_mask_vals[valid_mask] = distances_all <= distance_threshold

        # Optional: remove small islands and smooth
        mask = (ground_mask_vals.astype(np.uint8) * 255)
        try:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        except Exception:
            pass

        return mask

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


def main(args=None):
    rclpy.init(args=None)
    rd_node = LineDetector()
    rclpy.spin(rd_node)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
