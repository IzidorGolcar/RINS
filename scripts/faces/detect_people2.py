#!/usr/bin/env python3

import message_filters
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
import rclpy
import rclpy.duration
from rclpy.time import Time
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image, CompressedImage
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import tf2_geometry_msgs
import tf2_ros
from visualization_msgs.msg import Marker, MarkerArray


class FaceDetector(Node):
    MIN_DIST = 0.5
    MAX_DIST = 3.5
    FACE_MIN_SEPARATION = 0.8
    BLAZE_CONFIDENCE = 0.5
    MODEL_FILENAME = 'face_detection_short_range_with_metadata.tflite'
    CONFIRM_HITS = 2
    CANDIDATE_RADIUS = 0.4
    DEPTH_SAMPLE_RADIUS = 4

    def __init__(self):
        super().__init__('face_detector')

        self.bridge = CvBridge()
        self.candidates = []
        self.confirmed_faces = []

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.mp_face_detection = self._create_face_detector()

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.face_cascade_alt2 = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')

        self.declare_parameters('', [
            ('rgb_topic', '/gemini/color/image_raw/compressed'),
            ('depth_topic', '/gemini/depth/image_raw/compressedDepth'),
            ('camera_frame', 'gemini_color_frame'),
        ])

        rgb_topic = self.get_parameter('rgb_topic').value
        depth_topic = self.get_parameter('depth_topic').value
        self._camera_frame = self.get_parameter('camera_frame').value

        self.fx = self.fy = self.cx_p = self.cy_p = None
        # camera_info lives next to the raw image topic, not the compressed one.
        cam_info_base = rgb_topic.replace('/compressed', '').rsplit('/', 1)[0]
        cam_info_topic = cam_info_base + '/camera_info'

        self.create_subscription(CameraInfo, cam_info_topic,
                                 self._cam_info_cb, 10)
        self.create_subscription(CameraInfo, cam_info_topic,
                                 self._cam_info_cb, qos_profile_sensor_data)

        rgb_sub = message_filters.Subscriber(
            self, CompressedImage, rgb_topic, qos_profile=qos_profile_sensor_data)
        depth_sub = message_filters.Subscriber(
            self, CompressedImage, depth_topic, qos_profile=qos_profile_sensor_data)

        # FIXED: more stable sync settings
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub],
            queue_size=20,
            slop=0.5,
            allow_headerless=True
        )
        self.ts.registerCallback(self.synced_callback)

        self.marker_pub = self.create_publisher(
            MarkerArray, '/people_markers', QoSReliabilityPolicy.BEST_EFFORT)

        self.get_logger().info("Face detector initialized")

    # ---------------- MEDIAPIPE ----------------

    def _create_face_detector(self) -> vision.FaceDetector:
        model_path = Path(__file__).resolve().parent / 'models' / self.MODEL_FILENAME

        base_options = python.BaseOptions(model_asset_path=str(model_path))
        options = vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=self.BLAZE_CONFIDENCE,
        )
        return vision.FaceDetector.create_from_options(options)

    # ---------------- CAMERA INFO ----------------

    def _cam_info_cb(self, msg: CameraInfo):
        if self.fx is None:
            self.fx = msg.k[0]
            self.fy = msg.k[4]
            self.cx_p = msg.k[2]
            self.cy_p = msg.k[5]

    # ---------------- MAIN CALLBACK ----------------

    def synced_callback(self, rgb_msg: CompressedImage, depth_msg: CompressedImage):

        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(rgb_msg, 'passthrough')

            # compressedDepth = 12-byte ConfigHeader + PNG payload; strip header, PNG-decode.
            depth_payload = np.frombuffer(bytes(depth_msg.data)[12:], dtype=np.uint8)
            raw_depth = cv2.imdecode(depth_payload, cv2.IMREAD_UNCHANGED)

            # ✅ SAFE CHECKS (FIXED CRASH ROOT CAUSE)
            if cv_image is None or raw_depth is None:
                self.get_logger().warn("Received a null RGB or depth frame")
                return

            if cv_image.size == 0 or raw_depth.size == 0:
                self.get_logger().warn("Received an empty RGB or depth frame")
                return

            if cv_image.ndim != 3 or cv_image.shape[2] != 3:
                self.get_logger().warn(
                    f"Unexpected RGB frame shape {cv_image.shape}")
                return

            if cv_image.shape[0] < 10 or cv_image.shape[1] < 10:
                self.get_logger().warn("Empty RGB frame")
                return

            if raw_depth.dtype == np.uint16:
                depth_image = raw_depth.astype(np.float32) / 1000.0
            else:
                depth_image = raw_depth.astype(np.float32)

            if depth_image.size == 0:
                self.get_logger().warn("Empty depth frame")
                return

        except Exception as e:
            self.get_logger().error(f"CV error: {e}")
            return

        if self.fx is None:
            return

        vis = cv_image.copy()

        # SAFE cvtColor
        rgb_image = cv_image.copy()
        if rgb_image is None or rgb_image.size == 0:
            return

        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        mp_result = self.mp_face_detection.detect(mp_image)
        detections = mp_result.detections or []

        for detection in detections:

            bbox = detection.bounding_box
            x1 = max(0, int(bbox.origin_x))
            y1 = max(0, int(bbox.origin_y))
            x2 = min(cv_image.shape[1], int(bbox.origin_x + bbox.width))
            y2 = min(cv_image.shape[0], int(bbox.origin_y + bbox.height))

            if x2 <= x1 or y2 <= y1:
                continue

            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            roi = gray[y1:y2, x1:x2]

            if roi.size == 0:
                continue

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            z_patch = depth_image[max(0, cy-4):cy+4, max(0, cx-4):cx+4]
            valid = z_patch[np.isfinite(z_patch) & (z_patch > 0)]

            if valid.size == 0:
                continue

            z = float(np.median(valid))

            if not (self.MIN_DIST <= z <= self.MAX_DIST):
                continue

            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow("Face Detection", vis)
        cv2.waitKey(1)


def main():
    rclpy.init()
    node = FaceDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
