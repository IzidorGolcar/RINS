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




class LineDetector(Node):

    def __init__(self):
        super().__init__("line_detection")

        self.bridge = CvBridge()
        self.ground_distance_threshold = 0.0001
        self.ground_min_normal_y = 0.55
        self.ground_min_y_over_z = 1.1
    
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


    def stream_callback(self, rgb_msg, depth_msg):
        # Convert ROS Image messages to OpenCV images
        rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
        depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='32FC1')
        
        # Make explicit copies to avoid modifying originals
        rgb_display = rgb_image.copy()
        depth_for_processing = depth_image.copy()

        if self.received_camera_info:
            mask = self._ground_mask(depth_for_processing)
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
                line_labels = self._line_mask(depth_for_processing, rgb_display, mask)
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
                else:
                    line_overlay = rgb_display.clip(0, 255).astype(np.uint8)
                    line_overlay[m > 0] = 0
                    cv2.imshow('Lines', line_overlay)

        cv2.waitKey(1)



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
