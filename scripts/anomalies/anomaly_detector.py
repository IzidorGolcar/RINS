#!/usr/bin/python3

import os

from attr import dataclass
import torch

import rclpy
import rclpy.duration
from rclpy.node import Node
from rclpy.time import Time
import cv2
import numpy as np

from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import Bool, String
from cv_bridge import CvBridgeError, CvBridge
from geometry_msgs.msg import PointStamped
from rclpy.qos import qos_profile_sensor_data
from visualization_msgs.msg import Marker, MarkerArray
import tf2_geometry_msgs  # noqa: F401  (registers PointStamped tf transformer)
import tf2_ros
from util import *
from segmentation_model import SegmentationModel

from ament_index_python.packages import get_package_share_directory
import os

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
        self.joint_state_sub = self.create_subscription(JointState, "/joint_states", self.joint_state_callback, qos_profile_sensor_data)
        self.detection_enabled_sub = self.create_subscription(Bool, "/anomaly_detector/enabled", self.detection_enabled_callback, 10)
        self.detection_enabled = False
        
        self.arm_command_pub = self.create_publisher(String, "/arm_command", 10)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        pkg_dir = get_package_share_directory('dis_tutorial3')
        model_path = os.path.join(pkg_dir, 'models', 'anomaly_detector.pt')

        self.segmentation_model = SegmentationModel(model_path, device="cuda")
        self.tiles = []
        self._latest_depth = None
        self._latest_depth_frame = None
        self._latest_depth_stamp = None
        self._latest_camera_frame = None
        self.fx = self.fy = None
        self.cx_principal = self.cy_principal = None
        self._arm_joint_positions: dict[str, float] = {}
        self._arm_prev_joint_positions: dict[str, float] = {}
        self._arm_stable_samples = 0
        self._arm_is_moving = True

        self.tile_marker_pub = self.create_publisher(MarkerArray, '/tile_markers', 10)

        # Per-tile screenshots go here so the inspection-report PDF can
        # embed them. Wiped on every node start so the previous run's
        # tiles don't leak into this run's report.
        self.tile_img_dir = '/tmp/anomaly_tiles'
        os.makedirs(self.tile_img_dir, exist_ok=True)
        for fname in os.listdir(self.tile_img_dir):
            fp = os.path.join(self.tile_img_dir, fname)
            if os.path.isfile(fp):
                try:
                    os.remove(fp)
                except OSError:
                    pass
        self.get_logger().info(
            f'Tile screenshots will be saved under {self.tile_img_dir}/')

        cv2.namedWindow('Tile Detections', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Tiles', cv2.WINDOW_NORMAL)


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

    def joint_state_callback(self, msg):
        arm_joint_names = {
            'arm_base_joint',
            'arm_shoulder_joint',
            'arm_elbow_joint',
            'arm_wrist_joint',
        }

        joint_positions = {
            name: position
            for name, position in zip(msg.name, msg.position)
            if name in arm_joint_names
        }

        if len(joint_positions) != len(arm_joint_names):
            return

        if not self._arm_joint_positions:
            self._arm_joint_positions = joint_positions
            self._arm_prev_joint_positions = joint_positions.copy()
            self._arm_stable_samples = 0
            self._arm_is_moving = True
            return

        max_delta = max(
            abs(joint_positions[name] - self._arm_joint_positions[name])
            for name in arm_joint_names
        )

        self._arm_prev_joint_positions = self._arm_joint_positions.copy()
        self._arm_joint_positions = joint_positions

        if max_delta < 0.01:
            self._arm_stable_samples += 1
        else:
            self._arm_stable_samples = 0

        self._arm_is_moving = self._arm_stable_samples < 10

    def detection_enabled_callback(self, msg):
        self.detection_enabled = msg.data
        if self.detection_enabled:
            self.get_logger().info("Anomaly detection enabled")
            arm_cmd = String()
            arm_cmd.data = "look_at_belt_left"
            self.arm_command_pub.publish(arm_cmd)
        else:
            self.get_logger().info("Anomaly detection disabled")
            arm_cmd = String()
            arm_cmd.data = "garage"
            self.arm_command_pub.publish(arm_cmd)

    def _display_detection_disabled(self):
        """Display a black window with white text saying detection is disabled."""
        disabled_img = np.zeros((480, 640, 3), dtype=np.uint8)
        text = "Detection Disabled"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        thickness = 3
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (disabled_img.shape[1] - text_size[0]) // 2
        text_y = (disabled_img.shape[0] + text_size[1]) // 2
        cv2.putText(disabled_img, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
        cv2.imshow('Tile Detections', disabled_img)
        cv2.imshow('Tiles', disabled_img)

    def camera_frame_callback(self, msg):
        if not self.detection_enabled:
            self._display_detection_disabled()
            cv2.waitKey(1)
            return

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

    def _sample_tile_position(self, quad, mask):
        if self._latest_depth is None or self.fx is None or self._arm_is_moving:
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

    def _save_tile_screenshots(self, idx: int, tile) -> None:
        """Save the rectified tile and its anomaly overlay to disk so
        inspection_report can embed both in the PDF. Filenames carry
        the marker id so the report can look them up by tile id.
        """
        try:
            raw_path = os.path.join(self.tile_img_dir, f'tile_{idx}.png')
            cv2.imwrite(raw_path, tile.img)
            # Build an anomaly overlay: red where the segmentation mask
            # is positive, plus a green/red status banner.
            overlay = tile.img.copy()
            mask = tile.anomaly
            if mask is not None and mask.any():
                if mask.dtype == bool:
                    overlay[mask] = [0, 0, 255]
                else:
                    overlay[mask > 0] = [0, 0, 255]
            label = 'Anomaly Detected' if tile.is_anomalous else 'No Anomaly'
            label_colour = (0, 0, 255) if tile.is_anomalous else (0, 200, 0)
            cv2.putText(overlay, label, (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_colour, 2)
            overlay_path = os.path.join(
                self.tile_img_dir, f'tile_{idx}_anomaly.png')
            cv2.imwrite(overlay_path, overlay)
            self.get_logger().info(
                f'Saved tile {idx} screenshots: {raw_path}, {overlay_path}')
        except Exception as e:
            self.get_logger().warn(f'Failed to save tile {idx} images: {e}')

    def detect_anomalies(self, cv_image):
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
                # Save tile screenshots for the inspection report. Index
                # in the filename matches the marker `id` in
                # /tile_markers and the `TileEntry.id` in task2 →
                # inspection_report can embed by ID.
                idx = len(self.tiles) - 1
                self._save_tile_screenshots(idx, tile)
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


        else:
            cv2.polylines(detections, [quad], isClosed=True, color=(0, 0, 255), thickness=3)
            tile_display = np.zeros((304, 304, 3), dtype=np.uint8)         

        cv2.imshow('Tiles', tile_display)
        cv2.imshow('Tile Detections', detections)
        self._publish_tile_markers()
        


def main():

    rclpy.init(args=None)
    rd_node = AnomalyDetector()

    rclpy.spin(rd_node)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()