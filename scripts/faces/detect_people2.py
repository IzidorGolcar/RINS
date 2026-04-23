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
from rclpy.qos import (QoSDurabilityPolicy, QoSHistoryPolicy,
                       QoSProfile, QoSReliabilityPolicy,
                       qos_profile_sensor_data)
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
    BLAZE_CONFIDENCE = 0.4
    MODEL_FILENAME = 'face_detection_short_range_with_metadata.tflite'
    CONFIRM_HITS = 2
    CANDIDATE_RADIUS = 0.4
    DEPTH_SAMPLE_RADIUS = 6      # larger patch survives depth holes
    DEPTH_STD_MAX = 0.35         # tolerate noisier real-robot depth
    CANDIDATE_TIMEOUT_SEC = 2.5
    TARGET_FRAME = 'map'
    MARKER_QOS = QoSProfile(
        durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        reliability=QoSReliabilityPolicy.RELIABLE,
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=10)
    FRAME_SKIP = 1               # process every Nth frame (1 = all)
    INFER_SCALE = 3              # downsample for MediaPipe (3x = 9x speedup)
    USE_HAAR_FILTER = True       # run Haar on each MP bbox to cull wall/poster FPs

    def __init__(self):
        super().__init__('face_detector')

        self.bridge = CvBridge()
        self.candidates = []
        self.confirmed_faces = []
        self._next_face_id = 0
        self._frame_count = 0

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
            MarkerArray, '/people_markers', self.MARKER_QOS)

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
        # Drop every other frame (or more) when FRAME_SKIP > 1 to keep latency low.
        self._frame_count += 1
        if self._frame_count % self.FRAME_SKIP != 0:
            return

        # Decode JPEG directly — avoids cv_bridge's cvtColor2 quirks on Jazzy.
        try:
            rgb_payload = np.frombuffer(rgb_msg.data, dtype=np.uint8)
            cv_image = cv2.imdecode(rgb_payload, cv2.IMREAD_COLOR)

            depth_payload = np.frombuffer(bytes(depth_msg.data)[12:], dtype=np.uint8)
            raw_depth = cv2.imdecode(depth_payload, cv2.IMREAD_UNCHANGED)
        except Exception as e:
            self.get_logger().error(f'Frame decode failed: {e}')
            return

        if cv_image is None or raw_depth is None:
            return
        if cv_image.size == 0 or raw_depth.size == 0:
            return
        if cv_image.ndim != 3 or cv_image.shape[2] != 3:
            self.get_logger().warn(f'Unexpected RGB shape {cv_image.shape}')
            return

        depth_image = (raw_depth.astype(np.float32) / 1000.0
                       if raw_depth.dtype == np.uint16
                       else raw_depth.astype(np.float32))

        if self.fx is None:
            return

        vis = cv_image.copy()
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY) if self.USE_HAAR_FILTER else None

        # Align depth → RGB resolution once if drivers publish mismatched sizes.
        if depth_image.shape[:2] != cv_image.shape[:2]:
            depth_image = cv2.resize(
                depth_image,
                (cv_image.shape[1], cv_image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        # Downsample + BGR→RGB in one step for MediaPipe.
        s = self.INFER_SCALE
        proc = cv2.resize(cv_image, None, fx=1.0 / s, fy=1.0 / s,
                          interpolation=cv2.INTER_AREA)
        proc = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=proc)
        mp_result = self.mp_face_detection.detect(mp_image)
        detections = mp_result.detections or []

        new_positions = []
        for detection in detections:
            bbox = detection.bounding_box
            x1 = max(0, int(bbox.origin_x * s))
            y1 = max(0, int(bbox.origin_y * s))
            x2 = min(cv_image.shape[1], int((bbox.origin_x + bbox.width) * s))
            y2 = min(cv_image.shape[0], int((bbox.origin_y + bbox.height) * s))
            if x2 <= x1 or y2 <= y1:
                continue

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Haar secondary filter — rejects posters, wall textures, stickers
            # that MediaPipe sometimes hits on at low confidence.
            if self.USE_HAAR_FILTER:
                roi = gray[y1:y2, x1:x2]
                if not self._haar_has_face(roi):
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 180), 1)
                    continue

            # Multi-point depth sampling: median of 5 patches (centre + 4 off-centre)
            # is much more robust than a single patch that may land on a depth hole.
            r = self.DEPTH_SAMPLE_RADIUS
            face_w = x2 - x1
            face_h = y2 - y1
            probe_offsets = [
                (0, 0),
                (-face_w // 4, 0), (face_w // 4, 0),
                (0, -face_h // 4), (0, face_h // 4),
            ]
            samples = []
            for dx, dy in probe_offsets:
                px, py = cx + dx, cy + dy
                patch = depth_image[max(0, py - r):py + r + 1,
                                    max(0, px - r):px + r + 1]
                valid = patch[np.isfinite(patch) & (patch > 0)]
                if valid.size >= 3:
                    samples.append(float(np.median(valid)))
            if len(samples) < 2:
                continue

            samples_arr = np.array(samples)
            if float(np.std(samples_arr)) > self.DEPTH_STD_MAX:
                continue
            z = float(np.median(samples_arr))
            if not (self.MIN_DIST <= z <= self.MAX_DIST):
                continue

            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis, f'{z:.2f}m', (x1, max(y1 - 6, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

            pos_map = self._project_to_map(cx, cy, z, rgb_msg.header.stamp)
            if pos_map is not None:
                new_positions.append(pos_map)

        self._update_tracks(new_positions)
        self._publish_face_markers()

        cv2.putText(vis, f'confirmed: {len(self.confirmed_faces)}',
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Face Detection", vis)
        cv2.waitKey(1)

    def _haar_has_face(self, gray_roi):
        """Return True if either Haar cascade finds a face in the ROI."""
        if gray_roi is None or gray_roi.size == 0:
            return False
        h, w = gray_roi.shape[:2]
        if h < 16 or w < 16:
            # ROI too small for Haar to evaluate — trust MediaPipe.
            return True
        kwargs = dict(scaleFactor=1.1, minNeighbors=1, minSize=(12, 12))
        if len(self.face_cascade.detectMultiScale(gray_roi, **kwargs)) > 0:
            return True
        return len(self.face_cascade_alt2.detectMultiScale(gray_roi, **kwargs)) > 0

    # ---------------- TRACKING / PUBLISHING ----------------

    def _project_to_map(self, cx, cy, z, stamp):
        X_cam_opt = (cx - self.cx_p) * z / self.fx
        Y_cam_opt = (cy - self.cy_p) * z / self.fy

        pt = PointStamped()
        pt.header.frame_id = self._camera_frame
        pt.header.stamp = stamp
        pt.point.x = float(z)
        pt.point.y = float(-X_cam_opt)
        pt.point.z = float(-Y_cam_opt)

        stamp_time = Time.from_msg(stamp)
        # Try frame-time TF first (accurate when robot is rotating); fall back
        # to the latest TF so we don't pause localization whenever the camera
        # stamp drifts past the TF buffer's end.
        if self.tf_buffer.can_transform(
                self.TARGET_FRAME, self._camera_frame, stamp_time,
                timeout=rclpy.duration.Duration(seconds=0.05)):
            try:
                pt_map = self.tf_buffer.transform(
                    pt, self.TARGET_FRAME,
                    timeout=rclpy.duration.Duration(seconds=0.05))
                return np.array([pt_map.point.x, pt_map.point.y, pt_map.point.z])
            except Exception:
                pass

        pt.header.stamp = Time().to_msg()
        try:
            pt_map = self.tf_buffer.transform(
                pt, self.TARGET_FRAME,
                timeout=rclpy.duration.Duration(seconds=0.1))
        except Exception as e:
            self.get_logger().warn(
                f'TF transform failed (both stamp and latest): {e}',
                throttle_duration_sec=5.0)
            return None
        return np.array([pt_map.point.x, pt_map.point.y, pt_map.point.z])

    def _update_tracks(self, positions):
        now = self.get_clock().now()

        for pos in positions:
            # Match to a confirmed face — refresh and EMA-smooth its position.
            hit_confirmed = False
            for face in self.confirmed_faces:
                if np.linalg.norm(pos - face['pos']) < self.CANDIDATE_RADIUS:
                    face['pos'] = 0.8 * face['pos'] + 0.2 * pos
                    face['last_seen'] = now
                    hit_confirmed = True
                    break
            if hit_confirmed:
                continue

            # Match to a candidate — increment, promote when it hits CONFIRM_HITS.
            hit_candidate = False
            for cand in self.candidates:
                if np.linalg.norm(pos - cand['pos']) < self.CANDIDATE_RADIUS:
                    cand['pos'] = 0.5 * cand['pos'] + 0.5 * pos
                    cand['hits'] += 1
                    cand['last_seen'] = now
                    hit_candidate = True
                    if cand['hits'] >= self.CONFIRM_HITS:
                        too_close = any(
                            np.linalg.norm(cand['pos'] - f['pos']) < self.FACE_MIN_SEPARATION
                            for f in self.confirmed_faces
                        )
                        if not too_close:
                            self.confirmed_faces.append({
                                'pos': cand['pos'].copy(),
                                'id': self._next_face_id,
                                'last_seen': now,
                            })
                            self._next_face_id += 1
                        self.candidates.remove(cand)
                    break
            if hit_candidate:
                continue

            # New candidate — unless we're essentially on top of an existing face.
            too_close_to_confirmed = any(
                np.linalg.norm(pos - f['pos']) < self.FACE_MIN_SEPARATION
                for f in self.confirmed_faces
            )
            if not too_close_to_confirmed:
                self.candidates.append({
                    'pos': pos.copy(),
                    'hits': 1,
                    'last_seen': now,
                })

        # Drop candidates that stopped flashing — kills one-off false positives.
        timeout = rclpy.duration.Duration(seconds=self.CANDIDATE_TIMEOUT_SEC)
        self.candidates = [
            c for c in self.candidates if (now - c['last_seen']) < timeout
        ]

    def _publish_face_markers(self):
        if not self.confirmed_faces:
            return
        now = self.get_clock().now().to_msg()
        arr = MarkerArray()
        for face in self.confirmed_faces:
            m = Marker()
            m.header.frame_id = self.TARGET_FRAME
            m.header.stamp = now
            m.ns = 'faces'
            m.id = face['id']
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(face['pos'][0])
            m.pose.position.y = float(face['pos'][1])
            m.pose.position.z = float(face['pos'][2])
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.2
            m.color.r = 0.0
            m.color.g = 1.0
            m.color.b = 0.0
            m.color.a = 1.0
            arr.markers.append(m)
        self.marker_pub.publish(arr)


def main():
    rclpy.init()
    node = FaceDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
