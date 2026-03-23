#!/usr/bin/env python3

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
                ('detector_backend', 'haar'),
                ('yolo_model_path', 'yolo26n.pt'),
                ('yolo_confidence', 0.55),
                ('haar_scale_factor', 1.1),
                ('haar_min_neighbors', 6),
                ('haar_min_face_size', 20),
                ('min_depth_m', 0.25),
                ('max_depth_m', 4.0),
                ('target_frame', 'map'),
                ('merge_distance', 0.6),
                ('face_vote_threshold', 6),
        ])

        self.grid_resolution = 10.0
        self.meter_range = 10.0
        grid_size = int(self.meter_range * 2 * self.grid_resolution)
        self.face_position_votes = np.zeros((grid_size, grid_size))


        self.device = self.get_parameter('device').get_parameter_value().string_value
        self.detector_backend = self.get_parameter('detector_backend').get_parameter_value().string_value
        self.yolo_model_path = self.get_parameter('yolo_model_path').get_parameter_value().string_value
        self.yolo_confidence = self.get_parameter('yolo_confidence').get_parameter_value().double_value
        self.haar_scale_factor = self.get_parameter('haar_scale_factor').get_parameter_value().double_value
        self.haar_min_neighbors = self.get_parameter('haar_min_neighbors').get_parameter_value().integer_value
        self.haar_min_face_size = self.get_parameter('haar_min_face_size').get_parameter_value().integer_value
        self.min_depth_m = self.get_parameter('min_depth_m').get_parameter_value().double_value
        self.max_depth_m = self.get_parameter('max_depth_m').get_parameter_value().double_value
        self.target_frame = self.get_parameter('target_frame').get_parameter_value().string_value
        self.merge_distance = self.get_parameter('merge_distance').get_parameter_value().double_value
        self.face_vote_threshold = self.get_parameter('face_vote_threshold').get_parameter_value().integer_value
        self.bridge = CvBridge()
        self.faces = []
        self.detected_faces = []

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.rgb_image_sub = self.create_subscription(
            Image, "/oakd/rgb/preview/image_raw", self.rgb_callback, qos_profile_sensor_data)
        self.pointcloud_sub = self.create_subscription(
            PointCloud2, "/oakd/rgb/preview/depth/points", self.pointcloud_callback, qos_profile_sensor_data)

        marker_qos = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        self.marker_pub = self.create_publisher(Marker, "/people_marker", marker_qos)
        self.face_pub = self.create_publisher(PointStamped, '/detected_faces', 10)

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.model = YOLO(self.yolo_model_path)
        self.create_timer(2.0, self.publish_face_locations)

        self.get_logger().info("Node has been initialized!")

    def rgb_callback(self, data):
        current_faces = []
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

            if self.detector_backend == 'haar' and not self.face_cascade.empty():
                gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
                detections = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=float(self.haar_scale_factor),
                    minNeighbors=int(self.haar_min_neighbors),
                    minSize=(int(self.haar_min_face_size), int(self.haar_min_face_size)),
                )
                for (x, y, w, h) in detections:
                    if w <= 0 or h <= 0:
                        continue
                    aspect_ratio = float(w) / float(h)
                    if aspect_ratio < 0.65 or aspect_ratio > 1.45:
                        continue
                    current_faces.append((int(x + w / 2), int(y + h / 2)))
            else:
                res = self.model.predict(
                    cv_image,
                    imgsz=(256, 320),
                    show=False,
                    verbose=False,
                    classes=[0],
                    conf=float(self.yolo_confidence),
                    device=self.device,
                )

                for x in res:
                    boxes = x.boxes
                    if boxes is None or len(boxes) == 0:
                        continue
                    for bbox in boxes.xyxy:
                        cx = int((bbox[0] + bbox[2]) / 2)
                        cy = int((bbox[1] + bbox[3]) / 2)
                        current_faces.append((cx, cy))
            
            self.faces = current_faces
            cv2.imshow("image", cv_image)
            cv2.waitKey(1)
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge Error: {e}")

    def _is_new_face(self, x, y):
        for fx, fy, _ in self.detected_faces:
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
        peaks = (self.face_position_votes == local_max) & (self.face_position_votes >= self.face_vote_threshold)
        
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
                self.detected_faces.append((real_x, real_y, 1.5))
                self.get_logger().info(f"Detected face #{len(self.detected_faces)} at ({real_x:.2f}, {real_y:.2f})")

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
            if np.isnan(d[0]):
                continue
            if d[2] < self.min_depth_m or d[2] > self.max_depth_m:
                continue

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