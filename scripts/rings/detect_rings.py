#!/usr/bin/python3

import os
os.environ.setdefault('QT_LOGGING_RULES', 'default.warning=false;qt.qpa.*=false')

import rclpy
from rclpy.node import Node
import cv2, math
import numpy as np
import tf2_ros

from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped, Vector3, Pose
from cv_bridge import CvBridge, CvBridgeError
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, qos_profile_sensor_data
import message_filters
from geometry_msgs.msg import PointStamped
from rclpy.time import Time
import tf2_geometry_msgs
from sensor_msgs.msg import CameraInfo
from ring_map import *

qos_profile = QoSProfile(
          durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
          reliability=QoSReliabilityPolicy.RELIABLE,
          history=QoSHistoryPolicy.KEEP_LAST,
          depth=1)

class RingDetector(Node):
    def __init__(self):
        super().__init__('transform_point')

        self.bridge = CvBridge()

        self.declare_parameters('', [
            ('rgb_topic',    '/gemini/color/image_raw'),
            ('depth_topic',  '/gemini/depth/image_raw'),
            ('camera_frame', 'gemini_color_frame'),
        ])
        self._rgb_topic    = self.get_parameter('rgb_topic').get_parameter_value().string_value
        self._depth_topic  = self.get_parameter('depth_topic').get_parameter_value().string_value
        self._camera_frame = self.get_parameter('camera_frame').get_parameter_value().string_value

        self.rgb_sub = message_filters.Subscriber(
            self, Image, self._rgb_topic,
            qos_profile=qos_profile_sensor_data)
        self.depth_sub = message_filters.Subscriber(
            self, Image, self._depth_topic,
            qos_profile=qos_profile_sensor_data)

        self.stream = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=3,
            slop=0.2
        )

        self.stream.registerCallback(self.stream_callback)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.ring_pub = self.create_publisher(MarkerArray, '/ring_markers', qos_profile)
        self.marker_id = 0

        self.received_camera_info = False
        self.fx = self.fy = None
        self.cx_principal = self.cy_principal = None
        cam_info_topic = self._rgb_topic.rsplit('/', 1)[0] + '/camera_info'
        # Subscribe with BOTH QoS profiles — camera_info publisher can be
        # either RELIABLE or BEST_EFFORT depending on the driver build.
        self.cam_info_sub_reliable = self.create_subscription(
            CameraInfo, cam_info_topic, self.cam_info_callback, 10)
        self.cam_info_sub_sensor = self.create_subscription(
            CameraInfo, cam_info_topic, self.cam_info_callback,
            qos_profile_sensor_data)

        self.ring_map = RingMap()

        cv2.namedWindow('Detections', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Segmentation', cv2.WINDOW_NORMAL)

    def cam_info_callback(self, msg):
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


    def stream_callback(self, rgb_data, depth_data):
        try:
            cv_image  = self.bridge.imgmsg_to_cv2(rgb_data, 'bgr8')
            raw_depth = self.bridge.imgmsg_to_cv2(depth_data, desired_encoding='passthrough')
            if raw_depth.dtype == np.uint16:
                depth_image = raw_depth.astype(np.float32) / 1000.0
            else:
                depth_image = raw_depth.astype(np.float32)
        except Exception as e:
            self.get_logger().error(f'{e}')
            return

        if not self.received_camera_info:
            # Still show the raw feed so the user can see the stream is alive.
            cv2.imshow('Detections', cv_image)
            cv2.waitKey(1)
            self.get_logger().warn(
                'Waiting for camera_info – ring localization disabled.',
                throttle_duration_sec=5.0)
            return

        self.detect_rings(cv_image.copy(), depth_image.copy())
        cv2.waitKey(1)

    def estimate_height_from_ground(self, cy, avg_depth, img_h):
        H_cam = 1.05
        dy = cy - self.cy_principal
        h_rel = (dy * avg_depth) / self.fy
        absolute_height = H_cam - h_rel
        return absolute_height
    
    def get_roi(self, img_rgb, img_depth, max_depth=3.5):
        h, w = img_rgb.shape[:2]
        dist_mask = (img_depth > 0.1) & (img_depth <= max_depth)
        ground_cutoff = int(h * 0.6)
        no_ground_mask = np.ones((h, w), dtype=bool)
        no_ground_mask[ground_cutoff:, :] = False
        roi_mask = dist_mask & no_ground_mask
        foreground_rgb = np.zeros_like(img_rgb)
        foreground_rgb[roi_mask] = img_rgb[roi_mask]
        roi_pixels = foreground_rgb[roi_mask].reshape((-1, 3)).astype(np.float32)
        return roi_pixels, roi_mask

    def cluster_colors(self, roi_pixels, roi_mask, K=6):
        (h, w) = roi_mask.shape
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, _ = cv2.kmeans(roi_pixels, K, None, criteria, 8, cv2.KMEANS_RANDOM_CENTERS)
        labels_reshape = np.zeros((h, w), dtype=np.int32) - 1
        labels_reshape[roi_mask] = labels.flatten()
        return labels_reshape

    def build_label_map(self, labels):
        (h, w) = labels.shape
        label_map = np.zeros((h, w), dtype=np.uint8)
        current_id = 1

        K = len(np.unique(labels))

        for cluster_id in range(K):
            cluster_mask = (labels == cluster_id).astype(np.uint8) * 255
            num_labels, cc_labels = cv2.connectedComponents(cluster_mask)
            for i in range(1, num_labels):
                object_mask = (cc_labels == i)
                area = np.sum(object_mask)
                if 50 < area < (h * w * 0.1):
                    label_map[object_mask] = (current_id * 40) % 255
                    current_id += 1
        return label_map

    def display_label_map(self, label_map):
        unique_labels = np.unique(label_map)
        areas = {label: np.sum(label_map == label) for label in unique_labels if label != 0}
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
            label = f"RING"
            cv2.putText(output, label, (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, ring_color, 1)
        cv2.imshow('Detections', output)

    def is_grey(self, bgr: tuple[int, int, int], sat_threshold: int = 28) -> bool:
        pixel = np.uint8([[list(bgr)]])
        hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0, 0]
        s, v = int(hsv[1]), int(hsv[2])
        # Low saturation is grey/white/black. Very dark pixels are also noise.
        return s <= sat_threshold or v < 30

    def _is_hollow(self, img_depth, cx, cy, axes, avg_ring_depth,
                   gap_thresh: float = 0.12) -> bool:
        """Robust hollow check: the interior of a ring is either farther than
        the ring band or invalid (laser/IR passes through).
        Samples a patch half the minor axis wide around the centre."""
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
        # Either mostly invalid (laser passes through) or mostly farther.
        return (invalid + farther) / float(total) >= 0.6

    def find_rings(self, label_map, img_rgb, img_depth):
        results = []
        (h, w) = label_map.shape
        unique_labels = np.unique(label_map)

        # Minimum distance (pixels) from image border for a valid ring centre.
        border = 6

        for val in unique_labels:
            if val == 0: continue
            mask = (label_map == val).astype(np.uint8) * 255

            ring_color = self.get_average_color(img_rgb, mask)

            if self.is_grey(ring_color):
                continue

            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if len(cnt) < 8: continue

                cnt_area = cv2.contourArea(cnt)
                if cnt_area < 30:
                    continue

                ellipse = cv2.fitEllipse(cnt)
                (center, axes, _) = ellipse
                cx, cy = int(center[0]), int(center[1])

                # Reject shapes hugging the image border (incomplete).
                if cx < border or cy < border or cx >= w - border or cy >= h - border:
                    continue

                major, minor = max(axes), min(axes)
                if major <= 0:
                    continue

                aspect = minor / major
                if aspect < 0.40:
                    continue

                # Circularity: contour area vs ellipse area. A well-fit ring
                # has a ratio near 1; irregular blobs score much lower.
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

                # Require the ring band to have low depth spread. A flat poster
                # or the wall has a tight spread; noise/depth artefacts don't.
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

    def localize(self, rings):
        if self.fx is None:
            return

        target_frame = 'map'
        camera_frame = self._camera_frame

        # Skip silently until the map frame is available (AMCL still converging).
        if not self.tf_buffer.can_transform(
                target_frame, camera_frame, Time(),
                timeout=rclpy.duration.Duration(seconds=0.05)):
            self.get_logger().warn(
                f'TF {camera_frame} -> {target_frame} not yet available; '
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
            pt_cam.header.stamp = Time().to_msg()
            pt_cam.point.x = float(depth)
            pt_cam.point.y = float(-X_cam_opt)
            pt_cam.point.z = float(-Y_cam_opt)

            try:
                pt_world = self.tf_buffer.transform(
                    pt_cam, target_frame,
                    timeout=rclpy.duration.Duration(seconds=0.1)
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
            sphere.header.frame_id = 'map'
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
            label.header.frame_id = 'map'
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

    def detect_rings(self, img_rgb, img_depth):
        roi_pixels, roi_mask = self.get_roi(img_rgb, img_depth)
        clusters = self.cluster_colors(roi_pixels, roi_mask)
        label_map = self.build_label_map(clusters)
        self.display_label_map(label_map)
        rings = self.find_rings(label_map, img_rgb, img_depth)
        self.display_detections(img_rgb, rings)

        close_rings = [ring for ring in rings if ring['depth'] < 2]

        self.localize(close_rings)

        


def main():

    rclpy.init(args=None)
    rd_node = RingDetector()

    rclpy.spin(rd_node)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()