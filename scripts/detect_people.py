#!/usr/bin/env python3

import threading

import cv2
import numpy as np
import rclpy
import rclpy.duration
from rclpy.time import Time
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from rclpy.qos import QoSReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
import tf2_geometry_msgs  # noqa: F401 – registers PointStamped with tf2
import tf2_ros
from ultralytics import YOLO
from visualization_msgs.msg import Marker, MarkerArray


class FaceDetector(Node):

    # Tuning parameters

    # Min YOLO confidence to consider a detection
    CONFIDENCE_THRESH = 0.5

    # Euclidean distance from the camera (metres).
    MIN_DIST = 0.5
    MAX_DIST = 3.5

    # If a new candidate's map-frame position is within this radius (metres)
    # of an existing confirmed face it is treated as the same face and ignored.
    FACE_MIN_SEPARATION = 0.8

    # How many times a candidate must be "seen" before it is promoted to a confirmed face.
    CONFIRM_HITS = 1

    # Candidate detections within this radius (metres, map frame) are grouped
    # together as belonging to the same face candidate.
    CANDIDATE_RADIUS = 0.4

    # ------------------------------------------------------------------ #

    def __init__(self):
        super().__init__('face_detector')

        self.declare_parameters('', [('device', '')])
        self.device = self.get_parameter('device').get_parameter_value().string_value

        self.bridge = CvBridge()

        self.lock = threading.Lock()
        self.latest_detections: list[tuple[int, int]] = []  # (cx, cy) image coords

        # Candidate faces: each entry is {'pos': np.ndarray[3], 'count': int}
        self.candidates: list[dict] = []
        # Confirmed faces in map frame: list of np.ndarray([x, y, z])
        self.confirmed_faces: list[np.ndarray] = []

        # TF2
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Haar cascades – secondary filter to verify YOLO "person" hits are faces.
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.face_cascade_alt2 = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')

        self.rgb_sub = self.create_subscription(
            Image, '/oakd/rgb/preview/image_raw',
            self.rgb_callback, qos_profile_sensor_data)
        self.pc_sub = self.create_subscription(
            PointCloud2, '/oakd/rgb/preview/depth/points',
            self.pc_callback, qos_profile_sensor_data)

        self.marker_pub = self.create_publisher(
            MarkerArray, '/people_markers', QoSReliabilityPolicy.BEST_EFFORT)

        self.model = YOLO('yolov8n.pt')

        self.get_logger().info(
            'Face detector initialised. '
            f'confidence≥{self.CONFIDENCE_THRESH}, '
            f'depth [{self.MIN_DIST}–{self.MAX_DIST}] m, '
            f'confirm after {self.CONFIRM_HITS} hits.')



    def rgb_callback(self, data: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        except CvBridgeError as e:
            self.get_logger().error(str(e))
            return

        res = self.model.predict(
            cv_image,
            imgsz=(256, 320),
            show=False,
            verbose=False,
            classes=[0],              # 0 = person in COCO
            conf=self.CONFIDENCE_THRESH,
            device=self.device,
        )

        detections: list[tuple[int, int]] = []
        vis = cv_image.copy()

        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        for result in res:
            if result.boxes.xyxy.nelement() == 0:
                continue
            for idx in range(len(result.boxes.xyxy)):
                bbox = result.boxes.xyxy[idx].cpu().numpy()
                conf = float(result.boxes.conf[idx])

                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # Secondary filter: Haar cascade must also agree there is a face
                gray_roi = gray[y1:y2, x1:x2]
                if not self._haar_has_face(gray_roi):
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 180), 1)
                    continue

                detections.append((cx, cy))
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 220, 0), 2)
                cv2.circle(vis, (cx, cy), 5, (0, 220, 0), -1)
                cv2.putText(vis, f'{conf:.2f}', (x1, max(y1 - 6, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 0), 1)

        with self.lock:
            self.latest_detections = detections

        cv2.putText(vis, f'Confirmed faces: {len(self.confirmed_faces)}',
                    (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow('Face Detection', vis)
        if cv2.waitKey(1) == 27:
            rclpy.shutdown()

    def _haar_has_face(self, gray_roi: np.ndarray) -> bool:
        if gray_roi is None or gray_roi.size == 0:
            return False
        h, w = gray_roi.shape[:2]
        if h < 16 or w < 16:
            # Too small for Haar to be reliable – trust YOLO alone
            return True
        kwargs = dict(scaleFactor=1.1, minNeighbors=1, minSize=(12, 12))
        if len(self.face_cascade.detectMultiScale(gray_roi, **kwargs)) > 0:
            return True
        return len(self.face_cascade_alt2.detectMultiScale(gray_roi, **kwargs)) > 0

    def pc_callback(self, data: PointCloud2):
        with self.lock:
            detections = list(self.latest_detections)

        if not detections:
            return

        h, w = data.height, data.width
        pts = pc2.read_points_numpy(data, field_names=('x', 'y', 'z'))
        pts = pts.reshape((h, w, 3))

        for cx, cy in detections:
            if not (0 <= cx < w and 0 <= cy < h):
                continue

            d = pts[cy, cx, :]

            if np.any(np.isnan(d)) or np.all(d == 0.0):
                continue

            dist = float(np.linalg.norm(d))
            if not (self.MIN_DIST <= dist <= self.MAX_DIST):
                self.get_logger().debug(
                    f'Skipping detection: dist={dist:.2f} m outside '
                    f'[{self.MIN_DIST}, {self.MAX_DIST}] m')
                continue

            try:
                pt_cam = PointStamped()
                pt_cam.header.frame_id = data.header.frame_id
                pt_cam.header.stamp = Time(seconds=0).to_msg()
                pt_cam.point.x = float(d[0])
                pt_cam.point.y = float(d[1])
                pt_cam.point.z = float(d[2])

                pt_map = self.tf_buffer.transform(
                    pt_cam, 'map',
                    timeout=rclpy.duration.Duration(seconds=0.2))

                pos = np.array([pt_map.point.x, pt_map.point.y, pt_map.point.z])
            except Exception as exc:
                self.get_logger().warn(f'TF transform failed: {exc}')
                continue

            # TODO an actual face differentiator
            if any(
                np.linalg.norm(pos[:2] - f[:2]) < self.FACE_MIN_SEPARATION
                for f in self.confirmed_faces
            ):
                continue

            self._update_candidates(pos)

        self._publish_markers()

    def _update_candidates(self, pos: np.ndarray):

        best_idx: int | None = None
        best_dist = self.CANDIDATE_RADIUS

        for i, cand in enumerate(self.candidates):
            d = float(np.linalg.norm(pos[:2] - cand['pos'][:2]))
            if d < best_dist:
                best_dist = d
                best_idx = i

        if best_idx is not None:
            cand = self.candidates[best_idx]
            n = cand['count']
            cand['pos'] = (cand['pos'] * n + pos) / (n + 1)  # running average
            cand['count'] += 1

            if cand['count'] >= self.CONFIRM_HITS:
                face_id = len(self.confirmed_faces) + 1
                self.get_logger().info(
                    f'Face #{face_id} confirmed at '
                    f'({cand["pos"][0]:.2f}, {cand["pos"][1]:.2f}) m in map frame')
                self.confirmed_faces.append(cand['pos'].copy())
                self.candidates.pop(best_idx)
        else:
            self.candidates.append({'pos': pos.copy(), 'count': 1})


    def _publish_markers(self):
        if not self.confirmed_faces:
            return

        now = self.get_clock().now().to_msg()
        ma = MarkerArray()

        for i, pos in enumerate(self.confirmed_faces):
            # Sphere
            sphere = Marker()
            sphere.header.frame_id = 'map'
            sphere.header.stamp = now
            sphere.ns = 'faces'
            sphere.id = i
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position.x = float(pos[0])
            sphere.pose.position.y = float(pos[1])
            sphere.pose.position.z = float(pos[2])
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.3
            sphere.color.r = 1.0
            sphere.color.g = 0.4
            sphere.color.b = 0.0
            sphere.color.a = 1.0
            ma.markers.append(sphere)

            # Text label above the sphere
            label = Marker()
            label.header.frame_id = 'map'
            label.header.stamp = now
            label.ns = 'face_labels'
            label.id = i
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x = float(pos[0])
            label.pose.position.y = float(pos[1])
            label.pose.position.z = float(pos[2]) + 0.45
            label.pose.orientation.w = 1.0
            label.scale.z = 0.22
            label.color.r = label.color.g = label.color.b = 1.0
            label.color.a = 1.0
            label.text = f'Face {i + 1}'
            ma.markers.append(label)

        self.marker_pub.publish(ma)




def main():
    print('Face detection node starting.')
    rclpy.init(args=None)
    node = FaceDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
