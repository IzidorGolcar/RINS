#!/usr/bin/env python3

import message_filters
import importlib

import cv2
import numpy as np
import rclpy
import rclpy.duration
from rclpy.time import Time
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from rclpy.qos import (QoSDurabilityPolicy, QoSHistoryPolicy,
                       QoSProfile, QoSReliabilityPolicy,
                       qos_profile_sensor_data)
from sensor_msgs.msg import CameraInfo, CompressedImage
from nav_msgs.msg import OccupancyGrid
import tf2_geometry_msgs  # noqa: F401  (registers PointStamped transform)
import tf2_ros
from ultralytics import YOLO
from visualization_msgs.msg import Marker, MarkerArray


class FaceDetector(Node):
    CONFIDENCE_THRESH   = 0.24  # minimum YOLO confidence
    MIN_DIST            = 0.07   # metres – discard closer detections
    MAX_DIST            = 6.5   # metres – discard farther detections
    FACE_MIN_HEIGHT_M   = 0.0   # metres – map-frame Z lower bound
    FACE_MAX_HEIGHT_M   = 1.1   # metres – allow varied face sizes/heights
    FACE_MIN_SEPARATION = 0.35  # metres – same face if closer than this
    CONFIRM_HITS        = 1     # detections needed before face is confirmed
    CANDIDATE_RADIUS    = 0.6   # metres – cluster radius for candidates
    DEPTH_SAMPLE_RADIUS = 6     # pixels – neighbourhood radius for depth
    DEPTH_STD_MAX       = 0.45  # metres – tolerate noisier depth at range
    TARGET_FRAME        = 'map'

    MARKER_QOS = QoSProfile(
        durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        reliability=QoSReliabilityPolicy.RELIABLE,
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=10)

    def __init__(self):
        super().__init__('face_detector')

        self.declare_parameters('', [
            ('device', 'cuda:0'),
            ('require_gpu', True),
            ('target_fps', 15.0),
            ('show_debug', True),
            ('sync_queue_size', 20),
            ('sync_slop', 0.5),
            ('rgb_topic', '/gemini/color/image_raw/compressed'),
            ('depth_topic', '/gemini/depth/image_raw/compressedDepth'),
            ('camera_frame', 'gemini_color_frame'),
        ])
        self.device = self.get_parameter('device').get_parameter_value().string_value
        self.require_gpu = self.get_parameter('require_gpu').get_parameter_value().bool_value
        self.target_fps = float(self.get_parameter('target_fps').get_parameter_value().double_value)
        self.show_debug = self.get_parameter('show_debug').get_parameter_value().bool_value
        self.sync_queue_size = int(self.get_parameter('sync_queue_size').get_parameter_value().integer_value)
        self.sync_slop = float(self.get_parameter('sync_slop').get_parameter_value().double_value)
        rgb_topic = self.get_parameter('rgb_topic').get_parameter_value().string_value
        depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        self._camera_frame = self.get_parameter('camera_frame').get_parameter_value().string_value

        self._configure_inference_device()
        self._min_infer_period_ns = int(1e9 / max(self.target_fps, 1.0))
        self._last_infer_ns = 0

        self.bridge = CvBridge()

        self.candidates: list[dict] = []
        self.confirmed_faces: list[np.ndarray] = []
        self._arena_polygon: np.ndarray | None = None

        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.fx = self.fy = self.cx_p = self.cy_p = None
        cam_info_base = rgb_topic.replace('/compressed', '').rsplit('/', 1)[0]
        cam_info_topic = cam_info_base + '/camera_info'
        self.create_subscription(CameraInfo, cam_info_topic,
                                 self._cam_info_cb, 10)
        self.create_subscription(CameraInfo, cam_info_topic,
                                 self._cam_info_cb, qos_profile_sensor_data)

        # Haar cascades – secondary filter
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.face_cascade_alt2 = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')

        rgb_sub = message_filters.Subscriber(
            self, CompressedImage, rgb_topic, qos_profile=qos_profile_sensor_data)
        depth_sub = message_filters.Subscriber(
            self, CompressedImage, depth_topic, qos_profile=qos_profile_sensor_data)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub],
            queue_size=max(2, self.sync_queue_size),
            slop=max(0.01, self.sync_slop),
            allow_headerless=True)
        self.ts.registerCallback(self.synced_callback)

        self.marker_pub = self.create_publisher(
            MarkerArray, '/people_markers', self.MARKER_QOS)

        map_qos = QoSProfile(
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1)
        self.create_subscription(OccupancyGrid, '/map', self._map_cb, map_qos)

        self._torch.set_grad_enabled(False)
        self.model = YOLO('yolov8n.pt')
        self.model.fuse()

        self.get_logger().info(
            'Face detector initialised. '
            f'confidence≥{self.CONFIDENCE_THRESH}, '
            f'depth [{self.MIN_DIST}–{self.MAX_DIST}] m, '
            f'confirm after {self.CONFIRM_HITS} hits. '
            f'device={self.device}, target_fps={self.target_fps:.1f}. '
            f'show_debug={self.show_debug}, '
            f'sync_queue_size={self.sync_queue_size}, sync_slop={self.sync_slop:.2f}. '
            f'rgb={rgb_topic}, depth={depth_topic}, frame={self._camera_frame}')

    def _configure_inference_device(self) -> None:
        """Select and validate YOLO inference device for real-time use."""
        try:
            torch = importlib.import_module('torch')
        except Exception as exc:
            raise RuntimeError(
                'PyTorch is required for YOLO inference but could not be imported.') from exc

        self._torch = torch
        requested = (self.device or '').strip().lower()
        cuda_available = self._torch.cuda.is_available()

        if requested in ('', 'gpu', 'cuda'):
            requested = 'cuda:0'

        if requested.startswith('cuda') and not cuda_available:
            msg = 'CUDA device requested but no GPU is available.'
            if self.require_gpu:
                raise RuntimeError(msg)
            self.get_logger().warn(f'{msg} Falling back to CPU.')
            requested = 'cpu'

        if self.require_gpu and not requested.startswith('cuda'):
            raise RuntimeError(
                f'require_gpu=true but inference device is "{requested}". '
                'Set device to "cuda:0" and ensure NVIDIA runtime is available.')

        self.device = requested

    def _cam_info_cb(self, msg: CameraInfo) -> None:
        if self.fx is None:
            self.fx = msg.k[0]
            self.fy = msg.k[4]
            self.cx_p = msg.k[2]
            self.cy_p = msg.k[5]

    def _map_cb(self, msg: OccupancyGrid) -> None:
        """Build convex-hull polygon of free cells — same arena the waypoints cover."""
        data = np.array(msg.data, dtype=np.int8).reshape(
            msg.info.height, msg.info.width)
        ys, xs = np.where(data == 0)
        if ys.size < 3:
            return
        res = msg.info.resolution
        ox = msg.info.origin.position.x
        oy = msg.info.origin.position.y
        wx = ox + xs.astype(np.float32) * res
        wy = oy + ys.astype(np.float32) * res
        pts = np.column_stack([wx, wy]).astype(np.float32)
        self._arena_polygon = cv2.convexHull(pts)

    ARENA_TOLERANCE_M = 0.15     # faces sit on walls — allow slight outside-of-hull error

    def _is_in_arena(self, x: float, y: float) -> bool:
        if self._arena_polygon is None:
            return True
        # measureDist=True returns signed distance: + inside, - outside (metres here).
        dist = cv2.pointPolygonTest(
            self._arena_polygon, (float(x), float(y)), True)
        return dist >= -self.ARENA_TOLERANCE_M

    def _decode_depth(self, depth_msg: CompressedImage) -> np.ndarray | None:
        # compressedDepth = 12-byte ConfigHeader + PNG payload.
        try:
            payload = np.frombuffer(bytes(depth_msg.data)[12:], dtype=np.uint8)
            raw = cv2.imdecode(payload, cv2.IMREAD_UNCHANGED)
        except Exception as exc:
            self.get_logger().warn(f'Depth decode failed: {exc}')
            return None
        if raw is None or raw.size == 0:
            return None
        if raw.dtype == np.uint16:
            return raw.astype(np.float32) / 1000.0
        return raw.astype(np.float32)

    def synced_callback(self, rgb_msg: CompressedImage,
                        depth_msg: CompressedImage) -> None:
        now_ns = self.get_clock().now().nanoseconds
        if (now_ns - self._last_infer_ns) < self._min_infer_period_ns:
            return
        self._last_infer_ns = now_ns

        # Decode JPEG directly — cv_bridge's compressed_imgmsg_to_cv2('bgr8')
        # can raise SystemError via cvtColor2 on Jazzy when the source format
        # isn't what it expects.
        try:
            rgb_payload = np.frombuffer(rgb_msg.data, dtype=np.uint8)
            cv_image = cv2.imdecode(rgb_payload, cv2.IMREAD_COLOR)
        except Exception as exc:
            self.get_logger().warn(f'RGB decode failed: {exc}')
            return

        if cv_image is None or cv_image.size == 0:
            return
        if cv_image.ndim != 3 or cv_image.shape[2] != 3:
            self.get_logger().warn(f'Unexpected RGB shape {cv_image.shape}')
            return

        depth_image = self._decode_depth(depth_msg)
        if depth_image is None or depth_image.size == 0:
            return

        if self.fx is None:
            # Waiting for camera_info; still show the feed so user sees progress.
            if self.show_debug:
                cv2.imshow('Face Detection', cv_image)
                cv2.waitKey(1)
            return

        res = self.model.predict(
            cv_image,
            imgsz=(384, 384),
            show=False,
            verbose=False,
            classes=[0],
            conf=self.CONFIDENCE_THRESH,
            device=self.device,
            half=self.device.startswith('cuda'),
            max_det=8,
        )

        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        vis  = cv_image.copy()

        # Scale factors for bbox → depth image if resolutions differ.
        h_rgb, w_rgb = cv_image.shape[:2]
        h_d, w_d = depth_image.shape[:2]
        sx = w_d / float(w_rgb)
        sy = h_d / float(h_rgb)

        for result in res:
            if result.boxes.xyxy.nelement() == 0:
                continue

            for idx in range(len(result.boxes.xyxy)):
                bbox = result.boxes.xyxy[idx].cpu().numpy()
                conf = float(result.boxes.conf[idx])

                x1, y1, x2, y2 = (int(bbox[0]), int(bbox[1]),
                                   int(bbox[2]), int(bbox[3]))
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                gray_roi = gray[y1:y2, x1:x2]
                if not self._haar_has_face(gray_roi):
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 180), 1)
                    continue

                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 220, 0), 2)
                cv2.circle(vis, (cx, cy), 5, (0, 220, 0), -1)
                cv2.putText(vis, f'{conf:.2f}', (x1, max(y1 - 6, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 0), 1)

                # Sample depth at the bbox centre in depth-image coordinates.
                dx_ = int(cx * sx)
                dy_ = int(cy * sy)
                r = self.DEPTH_SAMPLE_RADIUS
                patch = depth_image[max(0, dy_ - r):dy_ + r + 1,
                                    max(0, dx_ - r):dx_ + r + 1]
                valid = patch[np.isfinite(patch) & (patch > 0)]
                if valid.size < 5:
                    continue
                if float(np.std(valid)) > self.DEPTH_STD_MAX:
                    continue

                z = float(np.median(valid))
                if not (self.MIN_DIST <= z <= self.MAX_DIST):
                    continue

                pos = self._project_to_map(cx, cy, z, rgb_msg.header.stamp)
                if pos is None:
                    continue
                if not (self.FACE_MIN_HEIGHT_M <= pos[2] <= self.FACE_MAX_HEIGHT_M):
                    continue
                if not self._is_in_arena(pos[0], pos[1]):
                    continue

                if any(
                    np.linalg.norm(pos[:2] - f[:2]) < self.FACE_MIN_SEPARATION
                    for f in self.confirmed_faces
                ):
                    continue

                self._update_candidates(pos)

        self._publish_markers()

        if self.show_debug:
            cv2.putText(vis, f'Confirmed faces: {len(self.confirmed_faces)}',
                        (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow('Face Detection', vis)
            if cv2.waitKey(1) == 27:
                rclpy.shutdown()

    def _project_to_map(self, cx: int, cy: int, z: float,
                        stamp) -> np.ndarray | None:
        X_cam_opt = (cx - self.cx_p) * z / self.fx
        Y_cam_opt = (cy - self.cy_p) * z / self.fy

        pt = PointStamped()
        pt.header.frame_id = self._camera_frame
        pt.header.stamp = stamp
        # gemini_color_frame is REP-103 optical (x-right, y-down, z-forward);
        # re-map to body-style (x-forward, y-left, z-up) so tf2 into `map` is correct.
        pt.point.x = float(z)
        pt.point.y = float(-X_cam_opt)
        pt.point.z = float(-Y_cam_opt)

        stamp_time = Time.from_msg(stamp)
        if not self.tf_buffer.can_transform(
                self.TARGET_FRAME, self._camera_frame, stamp_time,
                timeout=rclpy.duration.Duration(seconds=0.1)):
            self.get_logger().warn(
                f'TF {self._camera_frame}->{self.TARGET_FRAME} not ready.',
                throttle_duration_sec=5.0)
            return None
        try:
            pt_map = self.tf_buffer.transform(
                pt, self.TARGET_FRAME,
                timeout=rclpy.duration.Duration(seconds=0.1))
        except Exception as exc:
            self.get_logger().warn(f'TF transform failed: {exc}',
                                   throttle_duration_sec=5.0)
            return None
        return np.array([pt_map.point.x, pt_map.point.y, pt_map.point.z])

    def _haar_has_face(self, gray_roi: np.ndarray) -> bool:
        if gray_roi is None or gray_roi.size == 0:
            return False
        h, w = gray_roi.shape[:2]
        if h < 16 or w < 16:
            return True
        kwargs = dict(scaleFactor=1.1, minNeighbors=1, minSize=(12, 12))
        if len(self.face_cascade.detectMultiScale(gray_roi, **kwargs)) > 0:
            return True
        return len(self.face_cascade_alt2.detectMultiScale(gray_roi, **kwargs)) > 0

    def _update_candidates(self, pos: np.ndarray) -> None:
        best_idx: int | None = None
        best_dist = self.CANDIDATE_RADIUS

        for i, cand in enumerate(self.candidates):
            d = float(np.linalg.norm(pos[:2] - cand['pos'][:2]))
            if d < best_dist:
                best_dist, best_idx = d, i

        if best_idx is not None:
            cand = self.candidates[best_idx]
            n = cand['count']
            cand['pos']   = (cand['pos'] * n + pos) / (n + 1)
            cand['count'] += 1

            if cand['count'] >= self.CONFIRM_HITS:
                face_id = len(self.confirmed_faces) + 1
                self.get_logger().info(
                    f'Face #{face_id} confirmed at '
                    f'({cand["pos"][0]:.2f}, {cand["pos"][1]:.2f}) m')
                self.confirmed_faces.append(cand['pos'].copy())
                self.candidates.pop(best_idx)
        else:
            self.candidates.append({'pos': pos.copy(), 'count': 1})

    def _publish_markers(self) -> None:
        if not self.confirmed_faces:
            return

        now = self.get_clock().now().to_msg()
        ma  = MarkerArray()

        for i, pos in enumerate(self.confirmed_faces):
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
