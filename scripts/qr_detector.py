from __future__ import annotations

import json
import os
import sys
from typing import Optional

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time as RclTime
from rclpy.duration import Duration

import message_filters
import tf2_ros
import tf2_geometry_msgs  # noqa: F401  (registers PointStamped tf)

from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import ColorRGBA, String
from geometry_msgs.msg import Vector3, Point
from visualization_msgs.msg import Marker, MarkerArray

try:
    from pyzbar import pyzbar
except ImportError:
    pyzbar = None

# Reuse the same intent matcher the dialogue node uses for voice STT, so
# QR sentences like "Detect anomalies in the green cell." parse correctly.
# All scripts get installed flat into lib/dis_tutorial3/, so a same-dir
# import works at runtime; during dev (--symlink-install) we need to add
# the dialogue script folder explicitly.
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, 'dialogue'), os.path.join(_HERE, '..', 'dialogue')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
try:
    from intent_matcher import classify, classify_qr  # noqa: E402
except ImportError:
    classify = classify_qr = None  # type: ignore


# Re-publish the same payload only if we haven't seen it for this many seconds.
DEDUP_SECONDS = 5.0
# Map-frame distance under which two detections count as the "same" QR.
SAME_QR_RADIUS = 0.5


class QRDetector(Node):
    def __init__(self) -> None:
        super().__init__('qr_detector')

        if pyzbar is None:
            self.get_logger().error(
                'pyzbar not installed — QR detection disabled. '
                'pip install --user --break-system-packages pyzbar')

        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self._intr: Optional[dict] = None

        rgb_sub = message_filters.Subscriber(self, Image,
                                             '/oakd/rgb/preview/image_raw')
        depth_sub = message_filters.Subscriber(self, Image,
                                               '/oakd/rgb/preview/depth')
        sync = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub], queue_size=1, slop=0.1)
        sync.registerCallback(self._stream_cb)
        # Keep refs alive.
        self._rgb_sub = rgb_sub
        self._depth_sub = depth_sub
        self._sync = sync

        self.create_subscription(
            CameraInfo, '/oakd/rgb/preview/camera_info',
            self._cam_info_cb, qos_profile_sensor_data)

        self.text_pub = self.create_publisher(String, '/qr/decoded', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/qr_markers', 10)

        cv2.namedWindow('qr_detector', cv2.WINDOW_NORMAL)

        # In-flight set of (payload, map_xyz, last_seen_ns).
        self._known: list[dict] = []
        # Stable marker id per payload (so RViz dedups updates).
        self._marker_id: dict[str, int] = {}

        self.get_logger().info(
            f'QR detector ready (pyzbar={"OK" if pyzbar else "MISSING"}).')

    def _cam_info_cb(self, msg: CameraInfo) -> None:
        if self._intr is not None:
            return
        self._intr = {'fx': msg.k[0], 'fy': msg.k[4],
                      'cx': msg.k[2], 'cy': msg.k[5]}
        self.get_logger().info(
            f'Intrinsics fx={self._intr["fx"]:.1f} cx={self._intr["cx"]:.1f}')

    def _stream_cb(self, rgb_msg: Image, depth_msg: Image) -> None:
        if pyzbar is None or self._intr is None:
            return
        try:
            rgb = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
            depth = self.bridge.imgmsg_to_cv2(depth_msg, '32FC1')
        except Exception as e:
            self.get_logger().error(f'cv_bridge: {e}')
            return

        vis = rgb.copy()

        # OAK-D preview is small (~320×240); a face-sized QR at 0.7 m is
        # ~50 px — right at pyzbar's recall floor. 2× upscale on a grayscale
        # copy roughly doubles the catch rate without slowing pipeline.
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        up = cv2.resize(gray, (gray.shape[1] * 2, gray.shape[0] * 2),
                        interpolation=cv2.INTER_CUBIC)
        codes = pyzbar.decode(up)
        for code in codes:
            text = code.data.decode('utf-8', errors='ignore').strip()
            if not text:
                continue
            # Coords come back at 2× scale — divide for the original image.
            x = code.rect.left // 2
            y = code.rect.top // 2
            w = code.rect.width // 2
            h = code.rect.height // 2
            cx_px = x + w // 2
            cy_px = y + h // 2
            map_xyz = self._project_to_map(cx_px, cy_px, depth,
                                           rgb_msg.header.frame_id)
            # Pre-classify just for the on-screen label.
            preview_intent = None
            if classify is not None:
                preview_intent = classify(text).intent
            if preview_intent is None and classify_qr is not None:
                preview_intent = classify_qr(text)
            label_str = f'[{preview_intent}] {text}' if preview_intent else text
            # Always draw — even if map projection fails.
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(vis, label_str, (x, max(15, y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            self._handle_detection(text, map_xyz)

        cv2.imshow('qr_detector', vis)
        cv2.waitKey(1)

    def _project_to_map(self, cx_px: int, cy_px: int,
                        depth: np.ndarray, frame_id: str) -> Optional[np.ndarray]:
        h, w = depth.shape[:2]
        if not (0 <= cx_px < w and 0 <= cy_px < h):
            return None
        # Sample a small patch to dodge a single NaN at the centre.
        r = 3
        patch = depth[max(0, cy_px - r):cy_px + r + 1,
                      max(0, cx_px - r):cx_px + r + 1]
        finite = patch[np.isfinite(patch) & (patch > 0.1) & (patch < 6.0)]
        if finite.size == 0:
            return None
        z = float(np.median(finite))

        K = self._intr
        X_opt = (cx_px - K['cx']) * z / K['fx']
        Y_opt = (cy_px - K['cy']) * z / K['fy']
        # Optical → REP-103 body: x=forward, y=left, z=up.
        body = np.array([z, -X_opt, -Y_opt])

        try:
            tf = self.tf_buffer.lookup_transform(
                'map', frame_id, RclTime(),
                timeout=Duration(seconds=0.1))
        except Exception as e:
            self.get_logger().warn(f'TF map<-{frame_id}: {e}',
                                   throttle_duration_sec=2.0)
            return None
        q = tf.transform.rotation
        t = tf.transform.translation
        R = np.array([
            [1 - 2 * (q.y * q.y + q.z * q.z),
             2 * (q.x * q.y - q.z * q.w),
             2 * (q.x * q.z + q.y * q.w)],
            [2 * (q.x * q.y + q.z * q.w),
             1 - 2 * (q.x * q.x + q.z * q.z),
             2 * (q.y * q.z - q.x * q.w)],
            [2 * (q.x * q.z - q.y * q.w),
             2 * (q.y * q.z + q.x * q.w),
             1 - 2 * (q.x * q.x + q.y * q.y)],
        ])
        return R @ body + np.array([t.x, t.y, t.z])

    def _handle_detection(self, text: str, map_xyz: Optional[np.ndarray]) -> None:
        now_ns = self.get_clock().now().nanoseconds

        # Classify the QR text using the same intent matcher dialogue uses.
        intent: Optional[str] = None
        if classify is not None:
            parsed = classify(text)
            intent = parsed.intent
        if intent is None and classify_qr is not None:
            intent = classify_qr(text)

        # Dedup by text + nearby position.
        for k in self._known:
            same_text = k['text'] == text
            close = (map_xyz is not None and k['xyz'] is not None
                     and np.linalg.norm(map_xyz[:2] - k['xyz'][:2]) < SAME_QR_RADIUS)
            if same_text and close:
                if now_ns - k['last_ns'] < int(DEDUP_SECONDS * 1e9):
                    return  # already published recently
                k['last_ns'] = now_ns
                break
        else:
            self._known.append({'text': text, 'xyz': map_xyz, 'last_ns': now_ns})

        # Publish topic.
        payload = {'text': text,
                   'intent': intent,
                   'position': map_xyz.tolist() if map_xyz is not None else None}
        self.text_pub.publish(String(data=json.dumps(payload)))
        pos_str = (f' at ({map_xyz[0]:.2f}, {map_xyz[1]:.2f})'
                   if map_xyz is not None else ' (no depth)')
        intent_str = f' → {intent}' if intent else ' → (unrecognised)'
        self.get_logger().info(f'QR: {text!r}{intent_str}{pos_str}')

        # Publish marker (only if we have a map position).
        if map_xyz is None:
            return
        self._publish_marker(text, intent, map_xyz, now_ns)

    def _publish_marker(self, text: str, intent: Optional[str],
                        xyz: np.ndarray, now_ns: int) -> None:
        if text not in self._marker_id:
            self._marker_id[text] = len(self._marker_id)
        mid = self._marker_id[text]

        ma = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        cx, cy, cz = float(xyz[0]), float(xyz[1]), float(xyz[2])

        # 1. Square wireframe "frame" around the QR location — drawn in the
        #    map XY plane at the QR's z. QR is face-sized (~0.15-0.18 m) so
        #    half-side 0.08 m makes a 0.16 m square that matches reality.
        frame = Marker()
        frame.header.frame_id = 'map'
        frame.header.stamp = stamp
        frame.ns = 'qr_frame'
        frame.id = mid
        frame.type = Marker.LINE_STRIP
        frame.action = Marker.ADD
        frame.pose.orientation.w = 1.0
        frame.scale.x = 0.015  # line width
        frame.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
        s = 0.08
        for dx, dy in [(-s, -s), (s, -s), (s, s), (-s, s), (-s, -s)]:
            frame.points.append(Point(x=cx + dx, y=cy + dy, z=cz))
        frame.lifetime.sec = 0
        ma.markers.append(frame)

        # 2. Decoded text label, floating above the frame.
        label = Marker()
        label.header.frame_id = 'map'
        label.header.stamp = stamp
        label.ns = 'qr_text'
        label.id = mid
        label.type = Marker.TEXT_VIEW_FACING
        label.action = Marker.ADD
        label.pose.position.x = cx
        label.pose.position.y = cy
        label.pose.position.z = cz + 0.20
        label.pose.orientation.w = 1.0
        label.scale = Vector3(x=0.0, y=0.0, z=0.14)
        label.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
        # Show intent above the raw payload so RViz tells you both *what
        # was written* and *what task it parsed to*.
        if intent:
            label.text = f'[{intent}] {text}'
        else:
            label.text = text
        label.lifetime.sec = 0
        ma.markers.append(label)

        self.marker_pub.publish(ma)


def main() -> None:
    rclpy.init(args=None)
    node = QRDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
