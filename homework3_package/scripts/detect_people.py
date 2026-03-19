#!/usr/bin/env python3

import json
import os

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, QoSReliabilityPolicy
import rclpy.time

from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2 as pc2

from visualization_msgs.msg import Marker
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from scipy.ndimage import maximum_filter, label, center_of_mass
from tf2_ros import Buffer, TransformListener, TransformException
from tf2_geometry_msgs import do_transform_point
from ultralytics import YOLO
import math

class detect_faces(Node):

    def __init__(self):
        super().__init__('detect_faces')

        self.declare_parameters(
            namespace='',
            parameters=[
                ('device', 'cpu'),
                ('target_frame', 'map'),
                ('faces_file', os.path.expanduser('~/.ros/detected_faces.json')),
                ('merge_distance', 0.6),
        ])

        self.grid_resolution = 10.0
        self.meter_range = 10.0
        grid_size = int(self.meter_range * 2 * self.grid_resolution)
        self.face_position_votes = np.zeros((grid_size, grid_size))


        self.device = self.get_parameter('device').get_parameter_value().string_value
        self.target_frame = self.get_parameter('target_frame').get_parameter_value().string_value
        self.faces_file = os.path.expanduser(self.get_parameter('faces_file').get_parameter_value().string_value)
        self.merge_distance = self.get_parameter('merge_distance').get_parameter_value().double_value
        self.bridge = CvBridge()
        self.faces = []
        self.saved_faces = []
        self._load_saved_faces()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.rgb_image_sub = self.create_subscription(
            Image, "/oakd/rgb/preview/image_raw", self.rgb_callback, qos_profile_sensor_data)
        self.pointcloud_sub = self.create_subscription(
            PointCloud2, "/oakd/rgb/preview/depth/points", self.pointcloud_callback, qos_profile_sensor_data)

        marker_qos = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        self.marker_pub = self.create_publisher(Marker, "/people_marker", marker_qos)
        self.face_pub = self.create_publisher(PointStamped, '/detected_faces', 10)

        self.model = YOLO("yolo26n.pt") 
        self.create_timer(2.0, self.publish_face_locations)

        if self.saved_faces:
            self.create_timer(2.0, self._republish_saved_faces)

        self.get_logger().info("Node has been initialized!")

    def rgb_callback(self, data):
        current_faces = []
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            res = self.model.predict(cv_image, imgsz=(256, 320), show=False, verbose=False, classes=[0], device=self.device)

            for x in res:
                bbox = x.boxes.xyxy
                if bbox.nelement() == 0: continue
                bbox = bbox[0]
                cx, cy = int((bbox[0]+bbox[2])/2), int((bbox[1]+bbox[3])/2)
                current_faces.append((cx, cy))
            
            self.faces = current_faces
            cv2.imshow("image", cv_image)
            cv2.waitKey(1)
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge Error: {e}")

    def _load_saved_faces(self):
        if not os.path.exists(self.faces_file):
            return
        try:
            with open(self.faces_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for entry in data:
                self.saved_faces.append((float(entry['x']), float(entry['y']), float(entry.get('z', 0.0))))
            self.get_logger().info(f"Loaded {len(self.saved_faces)} saved face(s) from {self.faces_file}.")
        except Exception as e:
            self.get_logger().warn(f"Failed to load faces file: {e}")

    def _save_faces(self):
        folder = os.path.dirname(self.faces_file)
        if folder:
            os.makedirs(folder, exist_ok=True)
        payload = [{'x': x, 'y': y, 'z': z} for x, y, z in self.saved_faces]
        with open(self.faces_file, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)

    def _republish_saved_faces(self):
        for fx, fy, fz in self.saved_faces:
            msg = PointStamped()
            msg.header.frame_id = self.target_frame
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.point.x = fx
            msg.point.y = fy
            msg.point.z = fz
            self.face_pub.publish(msg)
        self.get_logger().info(f"Republished {len(self.saved_faces)} saved face(s) on startup.")

    def _is_new_face(self, x, y):
        for fx, fy, _ in self.saved_faces:
            if math.hypot(x - fx, y - fy) < self.merge_distance:
                return False
        return True

    def on_face_located(self, x, y):
        idx_x = int((x + self.meter_range) * self.grid_resolution)
        idx_y = int((y + self.meter_range) * self.grid_resolution)

        if 0 <= idx_x < self.face_position_votes.shape[0] and 0 <= idx_y < self.face_position_votes.shape[1]:
            self.face_position_votes[idx_x, idx_y] += 1

    def publish_face_locations(self):
        local_max = maximum_filter(self.face_position_votes, size=3)
        peaks = (self.face_position_votes == local_max) & (self.face_position_votes >= 5)
        
        labeled, num_features = label(peaks)
    
        locations = center_of_mass(self.face_position_votes, labeled, range(1, num_features + 1))

        for grid_x, grid_y in locations:
            real_x = (grid_x / self.grid_resolution) - self.meter_range
            real_y = (grid_y / self.grid_resolution) - self.meter_range

            msg = PointStamped()
            msg.header.frame_id = self.target_frame
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.point.x = float(real_x)
            msg.point.y = float(real_y)
            msg.point.z = 1.5
            self.face_pub.publish(msg)

            if self._is_new_face(real_x, real_y):
                self.saved_faces.append((real_x, real_y, 1.5))
                self._save_faces()
                self.get_logger().info(f"Saved face #{len(self.saved_faces)} at ({real_x:.2f}, {real_y:.2f})")

        if num_features > 0:
            self.get_logger().info(f'Published {num_features} precision-localized faces.')

        self.face_position_votes.fill(0)

    def pointcloud_callback(self, data):
        faces_to_process = list(self.faces) 
        if not faces_to_process: return

        try:
            cloud_array = pc2.read_points_numpy(data, field_names=("x", "y", "z"))
            cloud_array = cloud_array.reshape((data.height, data.width, 3))
        except Exception as e:
            return

        for u, v in faces_to_process:
            if not (0 <= u < data.width and 0 <= v < data.height): continue
            d = cloud_array[v, u, :]
            if np.isnan(d[0]): continue

            raw_pt = PointStamped()
            raw_pt.header = data.header
            raw_pt.point.x, raw_pt.point.y, raw_pt.point.z = map(float, d)

            try:
                transform = self.tf_buffer.lookup_transform(self.target_frame, data.header.frame_id, rclpy.time.Time())
                face_msg = do_transform_point(raw_pt, transform)
                
                marker = Marker()
                marker.header.frame_id = self.target_frame
                marker.header.stamp = data.header.stamp
                marker.type = Marker.SPHERE
                marker.id = int(u + v)
                marker.scale.x = marker.scale.y = marker.scale.z = 0.15
                marker.color.r, marker.color.a = 1.0, 1.0
                marker.pose.position = face_msg.point
                marker.pose.orientation.w = 1.0
                self.marker_pub.publish(marker)

                self.on_face_located(face_msg.point.x, face_msg.point.y)

            except TransformException:
                continue

def main():
    rclpy.init(args=None)
    node = detect_faces()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()