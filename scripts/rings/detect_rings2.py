#!/usr/bin/python3

import os
os.environ.setdefault('QT_LOGGING_RULES', 'default.warning=false;qt.qpa.*=false')

import cv2
import math
import numpy as np
import rclpy
import rclpy.duration
import tf2_ros
import tf2_geometry_msgs  # noqa: F401  — registers PointStamped transform
import message_filters

from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import (
    QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile,
    QoSReliabilityPolicy, qos_profile_sensor_data,
)

from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from geometry_msgs.msg import PointStamped, Vector3
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge, CvBridgeError

from color_segmentation import ObjectDetector
from ring_map import RingMap


marker_qos = QoSProfile(
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)


class RingDetector(Node):
    TARGET_FRAME = 'map'

    def __init__(self):
        super().__init__('ring_detector_v2')

        self.bridge = CvBridge()
        self.object_detector = ObjectDetector()

        self.declare_parameters('', [
            ('rgb_topic',    '/gemini/color/image_raw/compressed'),
            ('depth_topic',  '/gemini/depth/image_raw/compressedDepth'),
            ('camera_frame', 'gemini_color_frame'),
        ])
        self._rgb_topic    = self.get_parameter('rgb_topic').value
        self._depth_topic  = self.get_parameter('depth_topic').value
        self._camera_frame = self.get_parameter('camera_frame').value

        self.rgb_sub = message_filters.Subscriber(
            self, CompressedImage, self._rgb_topic,
            qos_profile=qos_profile_sensor_data)
        self.depth_sub = message_filters.Subscriber(
            self, CompressedImage, self._depth_topic,
            qos_profile=qos_profile_sensor_data)

        self.stream = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=5,
            slop=0.2,
        )
        self.stream.registerCallback(self.stream_callback)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.ring_pub = self.create_publisher(MarkerArray, '/ring_markers', marker_qos)

        self.received_camera_info = False
        self.fx = self.fy = None
        self.cx_principal = self.cy_principal = None

        # camera_info lives next to the raw image topic, not the compressed one.
        cam_info_base = self._rgb_topic.replace('/compressed', '').rsplit('/', 1)[0]
        cam_info_topic = cam_info_base + '/camera_info'

        self.cam_info_sub_reliable = self.create_subscription(
            CameraInfo, cam_info_topic, self.cam_info_callback, 10)
        self.cam_info_sub_sensor = self.create_subscription(
            CameraInfo, cam_info_topic, self.cam_info_callback,
            qos_profile_sensor_data)

        self.ring_map = RingMap()

        cv2.namedWindow('Detections', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Segmentation', cv2.WINDOW_NORMAL)

    def cam_info_callback(self, msg: CameraInfo):
        if self.fx is None:
            self.fx = msg.k[0]
            self.fy = msg.k[4]
            self.cx_principal = msg.k[2]
            self.cy_principal = msg.k[5]
            self.get_logger().info(
                f'Camera intrinsics loaded: fx={self.fx:.2f}, fy={self.fy:.2f}, '
                f'cx={self.cx_principal:.2f}, cy={self.cy_principal:.2f}'
            )
            self.received_camera_info = True

    # ---------------- MAIN CALLBACK ----------------

    def stream_callback(self, rgb_data: CompressedImage, depth_data: CompressedImage):
        self.get_logger().info('callback')
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(rgb_data, 'passthrough')

            depth_payload = np.frombuffer(bytes(depth_data.data)[12:], dtype=np.uint8)
            raw_depth = cv2.imdecode(depth_payload, cv2.IMREAD_UNCHANGED)
        except (CvBridgeError, Exception) as e:
            self.get_logger().error(f'Frame decode failed: {e}')
            return

        # ---------------- SAFE GUARDS ----------------
        if cv_image is None or not isinstance(cv_image, np.ndarray):
            self.get_logger().warn('Invalid RGB frame (None or wrong type)')
            return
        if cv_image.size == 0 or cv_image.shape[0] < 10 or cv_image.shape[1] < 10:
            self.get_logger().warn('Empty RGB frame received')
            return
        if cv_image.ndim != 3 or cv_image.shape[2] != 3:
            self.get_logger().warn(f'Unexpected RGB frame shape {cv_image.shape}')
            return
        if raw_depth is None or raw_depth.size == 0:
            self.get_logger().warn('Empty depth frame received')
            return

        if raw_depth.dtype == np.uint16:
            depth_image = raw_depth.astype(np.float32) / 1000.0
        else:
            depth_image = raw_depth.astype(np.float32)

        if not self.received_camera_info:
            cv2.imshow('Detections', cv_image)
            cv2.waitKey(1)
            self.get_logger().warn(
                'Waiting for camera_info – ring localization disabled.',
                throttle_duration_sec=5.0)
            return

        # Depth frame may differ in size from RGB on some drivers; align if so.
        if depth_image.shape[:2] != cv_image.shape[:2]:
            depth_image = cv2.resize(
                depth_image,
                (cv_image.shape[1], cv_image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        # self.detect_rings(cv_image.copy(), depth_image.copy(), rgb_data.header.stamp)
        cv2.imshow('Detections', cv_image)
        cv2.waitKey(1)

    # ---------------- GEOMETRY HELPERS ----------------

    def estimate_height_from_ground(self, cy, avg_depth, img_h):
        H_cam = 1.05
        dy = cy - self.cy_principal
        h_rel = (dy * avg_depth) / self.fy
        absolute_height = H_cam - h_rel
        return absolute_height

    def display_label_map(self, label_map):
        unique_labels = np.unique(label_map)
        areas = {label: np.sum(label_map == label)
                 for label in unique_labels if label != 0}
        sorted_labels = sorted(areas, key=lambda l: areas[l])
        rank_map = np.zeros_like(label_map, dtype=np.uint8)
        n = len(sorted_labels)
        for rank, label in enumerate(sorted_labels):
            color_idx = int(rank * 255 / max(n - 1, 1))
            rank_map[label_map == label] = color_idx
        colored_labels = cv2.applyColorMap(rank_map, cv2.COLORMAP_JET)
        colored_labels[label_map == 0] = 0
        cv2.imshow('Segmentation', colored_labels)

    def get_average_color(self, image_rgb, mask):
        mask_bool = mask.astype(bool)
        pixels = image_rgb[mask_bool]
        if pixels.size == 0:
            return (0, 0, 0)
        avg_color = pixels.mean(axis=0)
        return tuple(avg_color.astype(np.uint8).tolist())

    def display_detections(self, img_rgb, rings):
        output = img_rgb.copy()
        for ring in rings:
            ellipse = ring['ellipse']
            ring_color = ring['color']
            (center, _, _) = ellipse
            cx, cy = int(center[0]), int(center[1])
            cv2.ellipse(output, ellipse, ring_color, 2)
            cv2.putText(output, 'RING', (cx, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, ring_color, 1)
        cv2.putText(
            output,
            f'confirmed: {len(self.ring_map.confirmed_landmarks())}',
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
        )
        cv2.imshow('Detections', output)

    def _is_hollow(self, img_depth, cx, cy, axes, avg_ring_depth,
                   gap_thresh: float = 0.12) -> bool:
        """Hollow check: the interior of a real ring is either farther than
        the ring band or invalid (IR passes through)."""
        h, w = img_depth.shape[:2]
        inner_r = max(2, int(min(axes) * 0.25))
        y0, y1 = max(0, cy - inner_r), min(h, cy + inner_r + 1)
        x0, x1 = max(0, cx - inner_r), min(w, cx + inner_r + 1)
        patch = img_depth[y0:y1, x0:x1]
        if patch.size == 0:
            return False
        invalid = np.sum(~np.isfinite(patch) | (patch <= 0.05))
        farther = np.sum(np.isfinite(patch) & (patch > avg_ring_depth + gap_thresh))
        total   = patch.size
        return (invalid + farther) / float(total) >= 0.6

    # ---------------- RING DETECTION ----------------

    def find_rings(self, label_map, img_rgb, img_depth):
        results = []
        (h, w) = label_map.shape
        unique_labels = np.unique(label_map)

        # Minimum distance (pixels) from image border for a valid ring centre.
        border = 6

        for val in unique_labels:
            if val == 0:
                continue
            mask = (label_map == val).astype(np.uint8) * 255

            ring_color = self.get_average_color(img_rgb, mask)

            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if len(cnt) < 8:
                    continue

                cnt_area = cv2.contourArea(cnt)
                if cnt_area < 30:
                    continue

                ellipse = cv2.fitEllipse(cnt)
                (center, axes, _) = ellipse
                cx, cy = int(center[0]), int(center[1])

                if cx < border or cy < border or cx >= w - border or cy >= h - border:
                    continue

                major, minor = max(axes), min(axes)
                if major <= 0:
                    continue

                aspect = minor / major
                if aspect < 0.40:
                    continue

                ellipse_area = math.pi * 0.25 * major * minor
                if ellipse_area <= 0:
                    continue
                circ = cnt_area / ellipse_area
                if not (0.55 < circ < 1.25):
                    continue

                if not (0 <= cy < h and 0 <= cx < w):
                    continue

                obj_depths = img_depth[mask > 0]
                valid_depths = obj_depths[np.isfinite(obj_depths) & (obj_depths > 0.05)]
                if valid_depths.size < 20:
                    continue
                avg_ring_depth = float(np.median(valid_depths))

                depth_std = float(np.std(valid_depths))
                if depth_std > 0.25:
                    continue

                physical_diameter = (major * avg_ring_depth) / self.fx

                if not self._is_hollow(img_depth, cx, cy, axes, avg_ring_depth):
                    continue

                height = self.estimate_height_from_ground(cy, avg_ring_depth, 240)
                if not (0.07 < physical_diameter < 0.35):
                    continue
                if not (1.35 < height < 1.85):
                    continue

                results.append({
                    'ellipse': ellipse,
                    'color': ring_color,
                    'depth': avg_ring_depth,
                })
        return results

    # ---------------- LOCALIZATION ----------------

    def localize(self, rings, stamp):
        if self.fx is None:
            return

        target_frame = self.TARGET_FRAME
        camera_frame = self._camera_frame
        stamp_time = Time.from_msg(stamp)

        # Use the actual frame-capture time so TF interpolates to that pose —
        # critical when the robot is rotating and the camera feed stutters.
        if not self.tf_buffer.can_transform(
                target_frame, camera_frame, stamp_time,
                timeout=rclpy.duration.Duration(seconds=0.1)):
            self.get_logger().warn(
                f'TF {camera_frame} -> {target_frame} not ready at frame time; '
                'ring localization paused.',
                throttle_duration_sec=5.0)
            return

        for ring in rings:
            (center, _, _) = ring['ellipse']
            cx_px, cy_px = center
            depth = ring['depth']

            X_cam_opt = (cx_px - self.cx_principal) * depth / self.fx
            Y_cam_opt = (cy_px - self.cy_principal) * depth / self.fy

            pt_cam = PointStamped()
            pt_cam.header.frame_id = camera_frame
            pt_cam.header.stamp = stamp
            pt_cam.point.x = float(depth)
            pt_cam.point.y = float(-X_cam_opt)
            pt_cam.point.z = float(-Y_cam_opt)

            try:
                pt_world = self.tf_buffer.transform(
                    pt_cam, target_frame,
                    timeout=rclpy.duration.Duration(seconds=0.1),
                )
            except Exception as e:
                self.get_logger().warn(
                    f'TF transform failed: {e}', throttle_duration_sec=5.0)
                continue

            pos = np.array([pt_world.point.x, pt_world.point.y, pt_world.point.z])
            self.ring_map.update(pos, ring['color'])

        self._publish_confirmed()

    def _publish_confirmed(self):
        confirmed = self.ring_map.confirmed_landmarks()

        now = self.get_clock().now().to_msg()
        marker_array = MarkerArray()

        for lm in confirmed:
            sphere = Marker()
            b, g, r = [c / 255.0 for c in lm.color]
            sphere.header.frame_id = self.TARGET_FRAME
            sphere.header.stamp = now
            sphere.ns = 'confirmed_rings'
            sphere.id = lm.id
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position.x = float(lm.position[0])
            sphere.pose.position.y = float(lm.position[1])
            sphere.pose.position.z = float(lm.position[2])
            sphere.pose.orientation.w = 1.0
            sphere.scale = Vector3(x=0.15, y=0.15, z=0.15)
            sphere.color = ColorRGBA(r=r, g=g, b=b, a=1.0)
            sphere.lifetime.sec = 0
            marker_array.markers.append(sphere)

            label = Marker()
            label.header.frame_id = self.TARGET_FRAME
            label.header.stamp = now
            label.ns = 'confirmed_rings_labels'
            label.id = lm.id
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x = float(lm.position[0])
            label.pose.position.y = float(lm.position[1])
            label.pose.position.z = float(lm.position[2]) + 0.25
            label.pose.orientation.w = 1.0
            label.scale = Vector3(x=0.0, y=0.0, z=0.15)
            label.color = ColorRGBA(r=r, g=g, b=b, a=1.0)
            label.text = f'Ring {lm.id}'
            marker_array.markers.append(label)

        if len(marker_array.markers) > 0:
            self.ring_pub.publish(marker_array)


    def detect_rings(self, img_rgb, img_depth, stamp):
        label_map = self.object_detector.get_labels(
            img_rgb,
            downscale_factor=2,
            n_clusters=9,
            sample_size=10_000,
            min_area=3200,
            morph_kernel_size=5,
            morph_iterations=2,
        )
        self.display_label_map(label_map)

        # rings = self.find_rings(label_map, img_rgb, img_depth)
        # self.display_detections(img_rgb, rings)

        # close_rings = [ring for ring in rings if ring['depth'] < 2]
        # self.localize(close_rings, stamp)


def main():
    rclpy.init(args=None)
    rd_node = RingDetector()
    try:
        rclpy.spin(rd_node)
    finally:
        cv2.destroyAllWindows()
        rd_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
