#!/usr/bin/env python3

import json
import os
import sys
import tempfile
from datetime import datetime, timezone

import message_filters

import cv2
import numpy as np
import rclpy
import rclpy.duration
from ament_index_python.packages import get_package_share_directory
from rclpy.time import Time
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from rclpy.qos import QoSReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from std_msgs.msg import String
import tf2_geometry_msgs  # noqa: F401  (registers PointStamped tf transformer)
import tf2_ros
from ultralytics import YOLO
from visualization_msgs.msg import Marker, MarkerArray

sys.path.insert(0, os.path.dirname(__file__))
from face_recognizer import Identity, PersonnelRecognizer  # noqa: E402


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

        self.declare_parameters('', [
            ('device', ''),
            ('personnel_dir', ''),
            ('store_path', os.path.expanduser('~/.ros/known_people.json')),
        ])
        self.device = self.get_parameter('device').get_parameter_value().string_value
        self.store_path = self.get_parameter('store_path').get_parameter_value().string_value

        self.bridge = CvBridge()

        self.candidates: list[dict] = []
        # Each confirmed face is: {'id': int, 'pos': np.ndarray, 'identity': Identity | None,
        #                          'first_seen': iso8601 str}
        self.confirmed_faces: list[dict] = []
        self._next_face_id = 1

        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        cascade_dir = cv2.data.haarcascades if hasattr(cv2, 'data') else '/usr/share/opencv4/haarcascades/'
        self.face_cascade = cv2.CascadeClassifier(
            os.path.join(cascade_dir, 'haarcascade_frontalface_default.xml'))
        self.face_cascade_alt2 = cv2.CascadeClassifier(
            os.path.join(cascade_dir, 'haarcascade_frontalface_alt2.xml'))

        self.recognizer = self._build_recognizer()

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
            slop=0.05)
        self.ts.registerCallback(self.synced_callback)

        self.marker_pub = self.create_publisher(
            MarkerArray, '/people_markers', QoSReliabilityPolicy.BEST_EFFORT)
        self.identity_pub = self.create_publisher(
            String, '/recognized_people', 10)

        self.model = YOLO('yolov8n.pt')

        self._load_persisted()

        self.get_logger().info(
            'Face detector initialised. '
            f'confidence≥{self.CONFIDENCE_THRESH}, '
            f'depth [{self.MIN_DIST}–{self.MAX_DIST}] m, '
            f'confirm after {self.CONFIRM_HITS} hits. '
            f'Personnel known: {len(self.recognizer.people) if self.recognizer else 0}.')

    # ------------------------------------------------------- recognizer setup

    def _build_recognizer(self) -> PersonnelRecognizer | None:
        personnel_dir = self.get_parameter('personnel_dir').get_parameter_value().string_value
        if not personnel_dir:
            try:
                share = get_package_share_directory('dis_tutorial3')
                personnel_dir = os.path.join(share, 'personnel')
            except Exception:
                personnel_dir = ''
        if not personnel_dir or not os.path.isdir(personnel_dir):
            # Last-ditch: source-tree relative to this script.
            candidate = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'personnel'))
            if os.path.isdir(candidate):
                personnel_dir = candidate

        if not personnel_dir or not os.path.isdir(personnel_dir):
            self.get_logger().warn(
                'No personnel directory found; running without face recognition.')
            return None

        try:
            rec = PersonnelRecognizer(personnel_dir)
            self.get_logger().info(f'Recogniser trained on {personnel_dir}')
            return rec
        except Exception as exc:
            self.get_logger().warn(f'Recogniser failed to initialise: {exc}')
            return None

    # ----------------------------------------------------------- main callback

    def synced_callback(self, rgb_msg: Image, pc_msg: PointCloud2) -> None:
        try:
            cv_image = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
        except CvBridgeError as e:
            self.get_logger().error(str(e))
            return

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

                gray_roi = gray[y1:y2, x1:x2]
                color_roi = cv_image[y1:y2, x1:x2]
                if not self._haar_has_face(gray_roi):
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 180), 1)
                    continue

                identity = self.recognizer.recognize(color_roi) if self.recognizer else None
                label_text = (
                    f'{identity.name} ({identity.role})' if identity is not None
                    else f'{conf:.2f}'
                )
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 220, 0), 2)
                cv2.circle(vis, (cx, cy), 5, (0, 220, 0), -1)
                cv2.putText(vis, label_text, (x1, max(y1 - 6, 0)),
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

                # If we already confirmed a face here, keep voting on its identity
                # (handy when LBPH locks in only after several frames).
                merged = self._merge_into_confirmed(pos, identity)
                if merged:
                    continue

                self._update_candidates(pos, identity)

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
            return True
        kwargs = dict(scaleFactor=1.1, minNeighbors=1, minSize=(12, 12))
        if len(self.face_cascade.detectMultiScale(gray_roi, **kwargs)) > 0:
            return True
        return len(self.face_cascade_alt2.detectMultiScale(gray_roi, **kwargs)) > 0

    # ------------------------------------------------------ candidate handling

    def _merge_into_confirmed(self, pos: np.ndarray, identity: Identity | None) -> bool:
        """If *pos* belongs to an already-confirmed face, refine and return True."""
        for face in self.confirmed_faces:
            if np.linalg.norm(pos[:2] - face['pos'][:2]) < self.FACE_MIN_SEPARATION:
                # Smooth position with a low-weight running update.
                face['pos'] = 0.85 * face['pos'] + 0.15 * pos
                votes = face.setdefault('votes', {})
                if identity is not None:
                    votes[identity.label_id] = votes.get(identity.label_id, 0.0) + identity.score
                else:
                    votes[-1] = votes.get(-1, 0.0) + 0.45
                self._refresh_identity(face)
                return True
        return False

    def _update_candidates(self, pos: np.ndarray, identity: Identity | None) -> None:
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
            if identity is not None:
                cand['votes'][identity.label_id] = (
                    cand['votes'].get(identity.label_id, 0.0) + identity.score)
            else:
                cand['votes'][-1] = cand['votes'].get(-1, 0.0) + 0.45

            if cand['count'] >= self.CONFIRM_HITS:
                face_id = self._next_face_id
                self._next_face_id += 1
                face = {
                    'id': face_id,
                    'pos': cand['pos'].copy(),
                    'votes': dict(cand['votes']),
                    'identity': None,
                    'first_seen': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                }
                self._refresh_identity(face)
                ident_str = (
                    f'{face["identity"].name} ({face["identity"].role})'
                    if face['identity'] else 'unknown'
                )
                self.get_logger().info(
                    f'Face #{face_id} confirmed at '
                    f'({face["pos"][0]:.2f}, {face["pos"][1]:.2f}) m – {ident_str}')
                self.confirmed_faces.append(face)
                self.candidates.pop(best_idx)
                self._persist_and_publish()
        else:
            cand = {'pos': pos.copy(), 'count': 1, 'votes': {}}
            if identity is not None:
                cand['votes'][identity.label_id] = identity.score
            else:
                cand['votes'][-1] = 0.45
            self.candidates.append(cand)

    def _refresh_identity(self, face: dict) -> None:
        """Pick the highest-voted label for *face* and update its `identity`."""
        votes: dict[int, float] = face.get('votes', {})
        if not votes or self.recognizer is None:
            return
        best_label = max(votes, key=lambda k: votes[k])
        
        previous = face.get('identity')
        if best_label == -1:
            face['identity'] = None
            if previous is not None:
                self._persist_and_publish()
            return

        meta = self.recognizer.people.get(best_label)
        if meta is None:
            return

        face['identity'] = Identity(
            label_id=best_label,
            name=meta['name'],
            role=meta['role'],
            gender=meta['gender'],
            score=float(votes[best_label] / max(1, sum(votes.values()))),
        )
        if previous is None or previous.label_id != best_label:
            self._persist_and_publish()

    # --------------------------------------------------------- persistence

    def _load_persisted(self) -> None:
        if not self.store_path or not os.path.exists(self.store_path):
            return
        try:
            with open(self.store_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as exc:
            self.get_logger().warn(f'Could not read {self.store_path}: {exc}')
            return

        for entry in data.get('faces', []):
            ident = None
            if entry.get('name') and self.recognizer is not None:
                # Look up label_id by name so future votes still merge.
                for lbl, meta in self.recognizer.people.items():
                    if meta['name'] == entry['name'] and meta['role'] == entry.get('role'):
                        ident = Identity(
                            label_id=lbl,
                            name=meta['name'],
                            role=meta['role'],
                            gender=meta['gender'],
                            score=float(entry.get('score', 0.0)),
                        )
                        break
            face = {
                'id': int(entry['id']),
                'pos': np.array([entry['x'], entry['y'], entry.get('z', 0.0)]),
                'votes': {} if ident is None else {ident.label_id: ident.score},
                'identity': ident,
                'first_seen': entry.get('first_seen',
                                        datetime.now(timezone.utc).isoformat(timespec='seconds')),
            }
            self.confirmed_faces.append(face)
            self._next_face_id = max(self._next_face_id, face['id'] + 1)

        if self.confirmed_faces:
            self.get_logger().info(
                f'Restored {len(self.confirmed_faces)} known faces from {self.store_path}')

    def _persist_and_publish(self) -> None:
        payload = {
            'faces': [
                {
                    'id': f['id'],
                    'name': f['identity'].name if f['identity'] else None,
                    'role': f['identity'].role if f['identity'] else None,
                    'gender': f['identity'].gender if f['identity'] else None,
                    'score': float(f['identity'].score) if f['identity'] else 0.0,
                    'x': float(f['pos'][0]),
                    'y': float(f['pos'][1]),
                    'z': float(f['pos'][2]),
                    'first_seen': f['first_seen'],
                }
                for f in self.confirmed_faces
            ]
        }
        text = json.dumps(payload, indent=2)

        msg = String()
        msg.data = text
        self.identity_pub.publish(msg)

        if not self.store_path:
            return
        try:
            os.makedirs(os.path.dirname(self.store_path), exist_ok=True)
            fd, tmp = tempfile.mkstemp(prefix='.known_people-', dir=os.path.dirname(self.store_path))
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                f.write(text)
            os.replace(tmp, self.store_path)
        except Exception as exc:
            self.get_logger().warn(f'Could not write {self.store_path}: {exc}')

    # ---------------------------------------------------------------- markers

    def _publish_markers(self) -> None:
        if not self.confirmed_faces:
            return

        now = self.get_clock().now().to_msg()
        ma  = MarkerArray()

        for face in self.confirmed_faces:
            i = face['id']
            pos = face['pos']
            ident = face['identity']
            label = (f'{ident.name} ({ident.role})'
                     if ident is not None else f'Face {i}')
            # Pink for she/her, cyan for he/him, neutral orange when unknown.
            if ident is None:
                colour = (1.0, 0.4, 0.0)
            elif ident.gender == 'female':
                colour = (1.0, 0.45, 0.75)
            else:
                colour = (0.0, 0.7, 1.0)

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
            sphere.color.r, sphere.color.g, sphere.color.b = colour
            sphere.color.a = 1.0
            ma.markers.append(sphere)

            text_marker = Marker()
            text_marker.header.frame_id = 'map'
            text_marker.header.stamp = now
            text_marker.ns = 'face_labels'
            text_marker.id = i
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.pose.position.x = float(pos[0])
            text_marker.pose.position.y = float(pos[1])
            text_marker.pose.position.z = float(pos[2]) + 0.45
            text_marker.pose.orientation.w = 1.0
            text_marker.scale.z = 0.22
            text_marker.color.r = text_marker.color.g = text_marker.color.b = 1.0
            text_marker.color.a = 1.0
            text_marker.text = label
            ma.markers.append(text_marker)

            # Machine-readable identity marker, JSON-encoded.
            meta_marker = Marker()
            meta_marker.header.frame_id = 'map'
            meta_marker.header.stamp = now
            meta_marker.ns = 'face_identities'
            meta_marker.id = i
            meta_marker.type = Marker.TEXT_VIEW_FACING
            meta_marker.action = Marker.ADD
            meta_marker.pose.position.x = float(pos[0])
            meta_marker.pose.position.y = float(pos[1])
            meta_marker.pose.position.z = float(pos[2]) + 0.7
            meta_marker.pose.orientation.w = 1.0
            meta_marker.scale.z = 0.001  # invisible by default; data is in `text`
            meta_marker.color.a = 0.0
            meta_marker.text = json.dumps({
                'id': i,
                'name': ident.name if ident else None,
                'role': ident.role if ident else None,
                'gender': ident.gender if ident else None,
            })
            ma.markers.append(meta_marker)

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
