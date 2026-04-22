#!/usr/bin/env python3

import message_filters
import cv2
import numpy as np
import rclpy
import rclpy.duration
from rclpy.time import Time
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from rclpy.qos import QoSReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
import tf2_geometry_msgs
import tf2_ros
from visualization_msgs.msg import Marker, MarkerArray


class FaceDetector(Node):
    MIN_DIST            = 0.5
    MAX_DIST            = 3.5
    FACE_MIN_SEPARATION = 0.8
    CONFIRM_HITS        = 2
    CANDIDATE_RADIUS    = 0.4
    DEPTH_SAMPLE_RADIUS = 4

    def __init__(self):
        super().__init__('face_detector')

        self.bridge = CvBridge()

        self.candidates: list[dict] = []
        self.confirmed_faces: list[np.ndarray] = []

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.face_cascade_alt2 = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')

        self.declare_parameters('', [
            ('rgb_topic', '/gemini/color/image_raw'),
            ('depth_topic', '/gemini/depth/image_raw'),
            ('camera_frame', 'gemini_color_frame'),
        ])

        rgb_topic = self.get_parameter('rgb_topic').get_parameter_value().string_value
        depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        self._camera_frame = self.get_parameter('camera_frame').get_parameter_value().string_value

        self.fx = self.fy = self.cx_p = self.cy_p = None

        cam_info_topic = rgb_topic.rsplit('/', 1)[0] + '/camera_info'

        self.create_subscription(CameraInfo, cam_info_topic, self._cam_info_cb, 10)
        self.create_subscription(CameraInfo, cam_info_topic, self._cam_info_cb, qos_profile_sensor_data)

        rgb_sub = message_filters.Subscriber(
            self, Image, rgb_topic, qos_profile=qos_profile_sensor_data)
        depth_sub = message_filters.Subscriber(
            self, Image, depth_topic, qos_profile=qos_profile_sensor_data)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub],
            queue_size=3,
            slop=0.2
        )
        self.ts.registerCallback(self.synced_callback)

        self.marker_pub = self.create_publisher(
            MarkerArray, '/people_markers', QoSReliabilityPolicy.BEST_EFFORT)

        self.get_logger().info('Face detector initialised.')

    def _cam_info_cb(self, msg: CameraInfo):
        if self.fx is None:
            self.fx = msg.k[0]
            self.fy = msg.k[4]
            self.cx_p = msg.k[2]
            self.cy_p = msg.k[5]

    def synced_callback(self, rgb_msg: Image, depth_msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
            raw_depth = self.bridge.imgmsg_to_cv2(depth_msg, 'passthrough')
        except CvBridgeError as e:
            self.get_logger().error(str(e))
            return

        # ---------------- SAFE GUARDS (FIX FOR cvtColor CRASH) ----------------
        if cv_image is None or not isinstance(cv_image, np.ndarray):
            self.get_logger().warn('Invalid RGB frame (None or wrong type)')
            return

        if cv_image.size == 0 or cv_image.shape[0] == 0 or cv_image.shape[1] == 0:
            self.get_logger().warn('Empty RGB frame received')
            return

        if raw_depth is None or raw_depth.size == 0:
            self.get_logger().warn('Empty depth frame received')
            return

        # depth conversion
        if raw_depth.dtype == np.uint16:
            depth_image = raw_depth.astype(np.float32) / 1000.0
        else:
            depth_image = raw_depth.astype(np.float32)

        # ensure camera info ready
        if self.fx is None:
            cv2.imshow('Face Detection', cv_image)
            cv2.waitKey(1)
            return

        # ---------------- SAFE CVT COLOR ----------------
        try:
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        except cv2.error as e:
            self.get_logger().warn(f'cvtColor failed: {e}')
            return

        vis = cv_image.copy()

        faces_1 = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(24, 24))
        faces_2 = self.face_cascade_alt2.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(24, 24))

        for x, y, w, h in list(faces_1) + list(faces_2):
            cx = x + w // 2
            cy = y + h // 2

            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 220, 0), 2)
            cv2.circle(vis, (cx, cy), 5, (0, 220, 0), -1)

            r = self.DEPTH_SAMPLE_RADIUS
            patch = depth_image[
                max(0, cy - r):cy + r + 1,
                max(0, cx - r):cx + r + 1
            ]

            valid = patch[np.isfinite(patch) & (patch > 0)]
            if valid.size == 0:
                continue

            z = float(np.median(valid))

            if not (self.MIN_DIST <= z <= self.MAX_DIST):
                continue

        cv2.imshow('Face Detection', vis)
        cv2.waitKey(1)

    def _update_candidates(self, pos: np.ndarray):
        best_idx = None
        best_dist = self.CANDIDATE_RADIUS

        for i, cand in enumerate(self.candidates):
            d = np.linalg.norm(pos[:2] - cand['pos'][:2])
            if d < best_dist:
                best_dist = d
                best_idx = i

        if best_idx is not None:
            cand = self.candidates[best_idx]
            n = cand['count']
            cand['pos'] = (cand['pos'] * n + pos) / (n + 1)
            cand['count'] += 1

            if cand['count'] >= self.CONFIRM_HITS:
                self.confirmed_faces.append(cand['pos'].copy())
                self.candidates.pop(best_idx)
        else:
            self.candidates.append({'pos': pos.copy(), 'count': 1})

    def _publish_markers(self):
        if not self.confirmed_faces:
            return

        ma = MarkerArray()
        now = self.get_clock().now().to_msg()

        for i, pos in enumerate(self.confirmed_faces):
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = now
            m.ns = 'faces'
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(pos[0])
            m.pose.position.y = float(pos[1])
            m.pose.position.z = float(pos[2])
            m.scale.x = m.scale.y = m.scale.z = 0.3
            m.color.r = 1.0
            m.color.g = 0.4
            m.color.b = 0.0
            m.color.a = 1.0
            ma.markers.append(m)

        self.marker_pub.publish(ma)


def main():
    rclpy.init()
    node = FaceDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()