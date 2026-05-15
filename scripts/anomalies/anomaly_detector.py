#!/usr/bin/python3

from attr import dataclass
import torch

import rclpy
from rclpy.node import Node
import cv2
import numpy as np

from sensor_msgs.msg import Image
from cv_bridge import CvBridgeError, CvBridge
from rclpy.qos import qos_profile_sensor_data
from util import *
from segmentation_model import SegmentationModel


@dataclass
class Tile:
    hash: np.ndarray
    img: np.ndarray
    anomaly: np.ndarray

    anomaly_area_threshold: int = 20

    @property
    def anomaly_area(self):
        return np.sum(self.anomaly)

    @property
    def is_anomalous(self):
        return self.anomaly_area >= self.anomaly_area_threshold


class AnomalyDetector(Node):
    
    def __init__(self):
        super().__init__('anomaly_detector')
        self.bridge = CvBridge()
        self.camera_sub = self.create_subscription(Image, "/top_camera/rgb/preview/image_raw", self.camera_frame_callback, qos_profile_sensor_data)

        self.segmentation_model = SegmentationModel("/home/izidor/ros2_ws/anomaly_detector.pt", device="cpu")
        self.tiles = []

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

            tile_img = rectify_tile(cv_image, quad)
            tile_hash = tile_phash(tile_img)

            if not any(tiles_are_same(tile_hash, h.hash) for h in self.tiles):
                print(f"Found New tile. All tiles {len(self.tiles) + 1}")
                prediction = self.segmentation_model.predict(tile_img)
                tile = Tile(hash=tile_hash, img=tile_img, anomaly=prediction)
                self.tiles.append(tile)
            else:
                tile = next(t for t in self.tiles if tiles_are_same(tile_hash, t.hash))
            
            tile_display = tile_img.copy()
            tile_display[tile.anomaly] = [0, 0, 255]
            if tile.is_anomalous:
                cv2.putText(tile_display, 'Anomaly Detected', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                cv2.putText(tile_display, f'Anomaly Area: {tile.anomaly_area} px', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            else:
                cv2.putText(tile_display, 'No Anomaly', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        else:
            cv2.polylines(detections, [quad], isClosed=True, color=(0, 0, 255), thickness=3)
            tile_display = np.zeros((512, 512, 3), dtype=np.uint8)         

        cv2.imshow('utility', tile_display)
        cv2.imshow('Tile Detections', detections)
        


def main():

    rclpy.init(args=None)
    rd_node = AnomalyDetector()

    rclpy.spin(rd_node)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()