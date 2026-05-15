#!/usr/bin/python3

from attr import dataclass
import torch

import rclpy
import rclpy.duration
from rclpy.node import Node
from rclpy.time import Time
import cv2
import numpy as np

from sensor_msgs.msg import CameraInfo, Image
from cv_bridge import CvBridgeError, CvBridge
from geometry_msgs.msg import PointStamped
from rclpy.qos import qos_profile_sensor_data
from visualization_msgs.msg import Marker, MarkerArray
import tf2_geometry_msgs  # noqa: F401  (registers PointStamped tf transformer)
import tf2_ros
from util import *
from segmentation_model import SegmentationModel
from nav_msgs.msg import Odometry

@dataclass
class Tile:
    hash: np.ndarray
    img: np.ndarray
    anomaly: np.ndarray
    position: np.ndarray | None = None

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
        self.depth_sub = self.create_subscription(Image, "/top_camera/rgb/preview/depth", self.depth_callback, qos_profile_sensor_data)
        self.camera_info_sub = self.create_subscription(CameraInfo, "/top_camera/rgb/preview/camera_info", self.camera_info_callback, qos_profile_sensor_data)
        self.odom_sub = self.create_subscription(Odometry, "/odom", self.odom_callback, qos_profile_sensor_data)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.segmentation_model = SegmentationModel("/home/izidor/ros2_ws/anomaly_detector_2.pt", device="cpu")
        self.tiles = []
        self._latest_depth = None
        self._latest_depth_frame = None
        self._latest_depth_stamp = None
        self._latest_camera_frame = None
        self.fx = self.fy = None
        self.cx_principal = self.cy_principal = None

        self.tile_marker_pub = self.create_publisher(MarkerArray, '/tile_markers', 10)

        cv2.namedWindow('Tile Detections', cv2.WINDOW_NORMAL)
        cv2.namedWindow('utility', cv2.WINDOW_NORMAL)


    def odom_callback(self, msg):
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation

    def camera_info_callback(self, msg):
        if self.fx is not None:
            return

        self._latest_camera_frame = msg.header.frame_id
        self.fx = float(msg.k[0])
        self.fy = float(msg.k[4])
        self.cx_principal = float(msg.k[2])
        self.cy_principal = float(msg.k[5])
        self.get_logger().info(
            f"Camera intrinsics loaded: fx={self.fx:.2f}, fy={self.fy:.2f}, "
            f"cx={self.cx_principal:.2f}, cy={self.cy_principal:.2f}"
        )

    def depth_callback(self, msg):
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except CvBridgeError as e:
            self.get_logger().error(f"Depth CV Bridge error: {e}")
            return

        if depth.dtype == np.uint16:
            depth = depth.astype(np.float32) / 1000.0
        else:
            depth = depth.astype(np.float32)

        self._latest_depth = depth
        self._latest_depth_frame = msg.header.frame_id
        self._latest_depth_stamp = msg.header.stamp

        
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

    def _sample_tile_position(self, quad, mask):
        if self._latest_depth is None or self.fx is None:
            return None

        depth_h, depth_w = self._latest_depth.shape[:2]
        center = quad.reshape(4, 2).mean(axis=0)

        rgb_h, rgb_w = mask.shape[:2]
        center_x = float(center[0])
        center_y = float(center[1])
        if rgb_w > 0 and rgb_h > 0 and (depth_w != rgb_w or depth_h != rgb_h):
            scale_x = depth_w / float(rgb_w)
            scale_y = depth_h / float(rgb_h)
            center_x *= scale_x
            center_y *= scale_y

        center_px = int(np.clip(round(center_x), 0, depth_w - 1))
        center_py = int(np.clip(round(center_y), 0, depth_h - 1))

        radius = 5
        patch = self._latest_depth[
            max(0, center_py - radius):min(depth_h, center_py + radius + 1),
            max(0, center_px - radius):min(depth_w, center_px + radius + 1),
        ]

        depth_pixels = patch[np.isfinite(patch) & (patch > 0)]
        if depth_pixels.size == 0:
            return None

        depth = float(np.median(depth_pixels))
        if depth <= 0:
            return None

        x_cam = (float(center[0]) - self.cx_principal) * depth / self.fx
        y_cam = (float(center[1]) - self.cy_principal) * depth / self.fy

        pt_cam = PointStamped()
        pt_cam.header.frame_id = self._latest_camera_frame or self._latest_depth_frame
        pt_cam.header.stamp = Time(seconds=0).to_msg()
        pt_cam.point.x = float(x_cam)
        pt_cam.point.y = float(y_cam)
        pt_cam.point.z = float(depth)

        try:
            pt_map = self.tf_buffer.transform(
                pt_cam, 'map', timeout=rclpy.duration.Duration(seconds=0.1)
            )
        except Exception as e:
            self.get_logger().warn(f"Tile TF transform failed: {e}")
            return None

        return np.array([pt_map.point.x, pt_map.point.y, pt_map.point.z], dtype=np.float32)

    def _publish_tile_markers(self):
        now = self.get_clock().now().to_msg()
        marker_array = MarkerArray()

        for idx, tile in enumerate(self.tiles):

            if tile.position is None:
                continue

            sphere = Marker()
            sphere.header.frame_id = 'map'
            sphere.header.stamp = now
            sphere.ns = 'tiles'
            sphere.id = idx
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position.x = float(tile.position[0])
            sphere.pose.position.y = float(tile.position[1])
            sphere.pose.position.z = float(tile.position[2])
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = 0.08
            sphere.scale.y = 0.08
            sphere.scale.z = 0.08
            sphere.color.a = 1.0
            sphere.color.r = 1.0 if tile.is_anomalous else 0.0
            sphere.color.g = 0.0 if tile.is_anomalous else 1.0
            sphere.color.b = 0.0
            marker_array.markers.append(sphere)

            label = Marker()
            label.header.frame_id = 'map'
            label.header.stamp = now
            label.ns = 'tiles_labels'
            label.id = idx
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x = float(tile.position[0])
            label.pose.position.y = float(tile.position[1])
            label.pose.position.z = float(tile.position[2] + 0.10)
            label.pose.orientation.w = 1.0
            label.scale.z = 0.08
            label.color.a = 1.0
            label.color.r = 1.0
            label.color.g = 1.0
            label.color.b = 1.0
            label.text = f"tile {idx + 1}"
            marker_array.markers.append(label)

        if marker_array.markers:
            self.tile_marker_pub.publish(marker_array)

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

            sampled_position = self._sample_tile_position(quad, mask)
            if sampled_position is not None:
                tile.position = sampled_position
            
            tile_display = tile_img.copy()
            tile_display[tile.anomaly] = [0, 0, 255]
            if tile.is_anomalous:
                cv2.putText(tile_display, 'Anomaly Detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                cv2.putText(tile_display, 'No Anomaly', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            if tile.position is not None:
                cv2.putText(
                    detections,
                    f"map x={tile.position[0]:.2f} y={tile.position[1]:.2f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2,
                )

        else:
            cv2.polylines(detections, [quad], isClosed=True, color=(0, 0, 255), thickness=3)
            tile_display = np.zeros((304, 304, 3), dtype=np.uint8)         

        cv2.imshow('utility', tile_display)
        cv2.imshow('Tile Detections', detections)
        self._publish_tile_markers()
        


def main():

    rclpy.init(args=None)
    rd_node = AnomalyDetector()

    rclpy.spin(rd_node)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()