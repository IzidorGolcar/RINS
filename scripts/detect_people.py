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
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
import tf2_geometry_msgs  # noqa: F401 – registers PointStamped with tf2
import tf2_ros
from ultralytics import YOLO
from visualization_msgs.msg import Marker, MarkerArray


class FaceDetector(Node):
    CONFIDENCE_THRESH   = 0.5   # minimum YOLO confidence
    MIN_DIST            = 0.5   # metres – discard closer detections
    MAX_DIST            = 3.5   # metres – discard farther detections
    FACE_MIN_SEPARATION = 0.8   # metres – same face if closer than this
    CONFIRM_HITS        = 2     # detections needed before face is confirmed
    CANDIDATE_RADIUS    = 0.4   # metres – cluster radius for candidates
    DEPTH_SAMPLE_RADIUS = 4     # pixels – neighbourhood radius for depth

    def __init__(self):
        super().__init__('face_detector')

        self.declare_parameters('', [('device', '')])
        self.device = self.get_parameter('device').get_parameter_value().string_value

        self.bridge = CvBridge()
        self.candidates:     list[dict]       = []
        self.confirmed_faces: list[np.ndarray] = []

        # TF2
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Haar cascades – secondary filter, two training sets for coverage
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.face_cascade_alt2 = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')

        rgb_sub = message_filters.Subscriber(
            self, Image,
            '/oakd/rgb/preview/image_raw',
            qos_profile=qos_profile_sensor_data)
        pc_sub = message_filters.Subscriber(
            self, PointCloud2,
            '/oakd/rgb/preview/depth/points',
            qos_profile=qos_profile_sensor_data)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, pc_sub],
            queue_size=5,
            slop=0.05)          # 50 ms tolerance – OAK-D streams are well synced
        self.ts.registerCallback(self.synced_callback)

        # Publisher
        self.marker_pub = self.create_publisher(
            MarkerArray, '/people_markers', QoSReliabilityPolicy.BEST_EFFORT)

        self.model = YOLO('yolov8n.pt')

        self.get_logger().info(
            'Face detector initialised. '
            f'confidence≥{self.CONFIDENCE_THRESH}, '
            f'depth [{self.MIN_DIST}–{self.MAX_DIST}] m, '
            f'confirm after {self.CONFIRM_HITS} hits.')


    def synced_callback(self, rgb_msg: Image, pc_msg: PointCloud2) -> None:
        try:
            cv_image = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
        except CvBridgeError as e:
            self.get_logger().error(str(e))
            return

        #yolo
        res = self.model.predict(
            cv_image,
            imgsz=(256, 320),
            show=False,
            verbose=False,
            classes=[0],
            conf=self.CONFIDENCE_THRESH,
            device=self.device,
        )

        h, w = pc_msg.height, pc_msg.width
        pts = pc2.read_points_numpy(pc_msg, field_names=('x', 'y', 'z'))
        pts = pts.reshape((h, w, 3))

        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        vis  = cv_image.copy()

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

                # Secondary filter, Haar
                gray_roi = gray[y1:y2, x1:x2]
                if not self._haar_has_face(gray_roi):
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 180), 1)
                    continue

                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 220, 0), 2)
                cv2.circle(vis, (cx, cy), 5, (0, 220, 0), -1)
                cv2.putText(vis, f'{conf:.2f}', (x1, max(y1 - 6, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 0), 1)

                r = self.DEPTH_SAMPLE_RADIUS
                patch = pts[max(0, cy - r):cy + r + 1,
                            max(0, cx - r):cx + r + 1, :]
                flat  = patch.reshape(-1, 3)
                valid_mask = (
                    ~np.isnan(flat).any(axis=1) &
                    ~(flat == 0).all(axis=1)
                )
                valid = flat[valid_mask]
                if valid.size == 0:
                    continue
                d = np.median(valid, axis=0)

                dist = float(np.linalg.norm(d))
                if not (self.MIN_DIST <= dist <= self.MAX_DIST):
                    self.get_logger().debug(
                        f'Skipping: dist={dist:.2f} m outside range')
                    continue

                try:
                    pt_cam = PointStamped()
                    pt_cam.header.frame_id = pc_msg.header.frame_id
                    # Time(0) = latest available transform (avoids
                    # extrapolation warnings from timestamp drift)
                    pt_cam.header.stamp = Time(seconds=0).to_msg()
                    pt_cam.point.x = float(d[0])
                    pt_cam.point.y = float(d[1])
                    pt_cam.point.z = float(d[2])

                    pt_map = self.tf_buffer.transform(
                        pt_cam, 'map',
                        timeout=rclpy.duration.Duration(seconds=0.2))

                    pos = np.array([
                        pt_map.point.x,
                        pt_map.point.y,
                        pt_map.point.z,
                    ])
                except Exception as exc:
                    self.get_logger().warn(f'TF transform failed: {exc}')
                    continue

                if any(
                    np.linalg.norm(pos[:2] - f[:2]) < self.FACE_MIN_SEPARATION
                    for f in self.confirmed_faces
                ):
                    continue

                self._update_candidates(pos)

        self._publish_markers()

        cv2.putText(vis, f'Confirmed faces: {len(self.confirmed_faces)}',
                    (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow('Face Detection', vis)
        if cv2.waitKey(1) == 27:
            rclpy.shutdown()

    def _haar_has_face(self, gray_roi: np.ndarray) -> bool:
        """Return True if either Haar cascade finds a face in *gray_roi*."""
        if gray_roi is None or gray_roi.size == 0:
            return False
        h, w = gray_roi.shape[:2]
        if h < 16 or w < 16:
            return True     # too small for Haar
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
