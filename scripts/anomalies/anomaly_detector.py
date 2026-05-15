#!/usr/bin/python3

import rclpy
from rclpy.node import Node
import cv2
import numpy as np

from sensor_msgs.msg import Image
from cv_bridge import CvBridgeError, CvBridge
from rclpy.qos import qos_profile_sensor_data
from util import generate_tile_mask, rectify_tile, is_tile_fully_visible

class AnomalyDetector(Node):
    
    def __init__(self):
        super().__init__('anomaly_detector')
        self.bridge = CvBridge()
        self.camera_sub = self.create_subscription(Image, "/top_camera/rgb/preview/image_raw", self.camera_frame_callback, qos_profile_sensor_data)
        cv2.namedWindow('Tile Detections', cv2.WINDOW_NORMAL)
        cv2.namedWindow('utility', cv2.WINDOW_NORMAL)

        
    def camera_frame_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.detect_anomalies(cv_image)
        except CvBridgeError as e:
            self.get_logger().error(f"CV Bridge error: {e}")
            return
        except Exception as e:
            self.get_logger().error(f"Unexpected error: {e}")
            return
        
        cv2.waitKey(1)


    def detect_anomalies(self, cv_image):
        self.detect_tiles(cv_image)

    def detect_tiles(self, cv_image):
        mask, quad = generate_tile_mask(cv_image)
        
        if quad is None:
            cv2.imshow('Tile Detections', cv_image)
            return
        
        detections = cv_image.copy()

        if is_tile_fully_visible(quad, cv_image.shape):
            cv2.polylines(detections, [quad], isClosed=True, color=(0, 255, 0), thickness=3)
            tile = rectify_tile(cv_image, quad)    
        else:
            cv2.polylines(detections, [quad], isClosed=True, color=(0, 0, 255), thickness=3)
            tile = np.zeros((512, 512, 3), dtype=np.uint8)

        cv2.imshow('utility', tile)
        cv2.imshow('Tile Detections', detections)
        


def main():

    rclpy.init(args=None)
    rd_node = AnomalyDetector()

    rclpy.spin(rd_node)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()