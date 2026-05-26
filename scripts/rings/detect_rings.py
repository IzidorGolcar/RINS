#!/usr/bin/python3

import rclpy
from rclpy.node import Node
import cv2, math
from typing import Optional
import numpy as np
import tf2_ros

from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped, Vector3, Pose
from cv_bridge import CvBridge, CvBridgeError
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
import message_filters
from geometry_msgs.msg import PointStamped
from rclpy.time import Time
import tf2_geometry_msgs
from sensor_msgs.msg import CameraInfo, Image
from ring_map import *

qos_profile = QoSProfile(
          durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
          reliability=QoSReliabilityPolicy.RELIABLE,
          history=QoSHistoryPolicy.KEEP_LAST,
          depth=1)

class RingDetector(Node):
    BLACK_MAX_V = 95
    BLACK_MAX_S = 80

    def __init__(self):
        super().__init__('transform_point')

        self.bridge = CvBridge()

        self.rgb_sub = message_filters.Subscriber(self, Image, "/oakd/rgb/preview/image_raw")
        self.depth_sub = message_filters.Subscriber(self, Image, "/oakd/rgb/preview/depth")

        self.stream = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=1,
            slop=0.1
        )

        self.stream.registerCallback(self.stream_callback)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.ring_pub = self.create_publisher(MarkerArray, '/ring_markers', qos_profile)
        self.marker_id = 0

        self.received_camera_info = False
        self.fx = self.fy = None
        self.cx_principal = self.cy_principal = None
        self.ground_mask_image = None
        self.ground_mask_sub = self.create_subscription(
            Image,
            '/line_detector/ground_mask',
            self.ground_mask_callback,
            qos_profile
        )
        self.cam_info_sub = self.create_subscription(
            CameraInfo,
            '/oakd/rgb/preview/camera_info',
            self.cam_info_callback,
            qos_profile
        )

        self.ring_map = RingMap()

        cv2.namedWindow('Ring Detections', cv2.WINDOW_NORMAL)

    def ground_mask_callback(self, msg):
        try:
            mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            self.ground_mask_image = (mask > 0).astype(np.uint8)
        except Exception as e:
            self.get_logger().warn(f'Failed to read ground mask: {e}')

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
        if not self.received_camera_info:
            return
        try:
            cv_image = self.bridge.imgmsg_to_cv2(rgb_data, 'bgr8')
            depth_image = self.bridge.imgmsg_to_cv2(depth_data, '32FC1')

            self.detect_rings(cv_image.copy(), depth_image.copy())

            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f'{e}')

    def estimate_height_from_ground(self, cy, avg_depth, img_h):
        H_cam = 1.05
        dy = cy - self.cy_principal
        h_rel = (dy * avg_depth) / self.fy
        absolute_height = H_cam - h_rel
        return absolute_height
    
    def get_roi(self, img_rgb, img_depth, ground_mask=None, max_depth=3.5):
        h, w = img_rgb.shape[:2]
        dist_mask = (img_depth > 0.1) & (img_depth <= max_depth)
        if ground_mask is not None and ground_mask.shape[:2] == (h, w):
            no_ground_mask = ground_mask <= 0
        else:
            print('no ground mask')
            ground_cutoff = int(h * 0.6)
            no_ground_mask = np.ones((h, w), dtype=bool)
            no_ground_mask[ground_cutoff:, :] = False
        roi_mask = dist_mask & no_ground_mask
        foreground_rgb = np.zeros_like(img_rgb)
        foreground_rgb[roi_mask] = img_rgb[roi_mask]
        roi_pixels = foreground_rgb[roi_mask].reshape((-1, 3)).astype(np.float32)
        return roi_pixels, roi_mask

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
        cv2.imshow('Ring Detections', output)

    def _classify_allowed_color(self, bgr: tuple[int, int, int]) -> Optional[str]:
        """Return an allowed ring colour or None for likely false positives."""
        b, g, r = [int(c) for c in bgr]
        px = np.uint8([[[b, g, r]]])
        h, s, v = cv2.cvtColor(px, cv2.COLOR_BGR2HSV)[0, 0]

        if (v <= self.BLACK_MAX_V and s <= self.BLACK_MAX_S) or (v < 55 and s < 140):
            return 'black'

        if s < 50:
            return None

        if h <= 12 or h >= 168:
            return 'red'
        if 40 <= h <= 90:
            return 'green'
        if 95 <= h <= 140:
            return 'blue'
        return None



    def find_rings(self, label_map, img_rgb, img_depth):
        results = []
        (h, w) = label_map.shape
        unique_labels = np.unique(label_map)
        hsv_image = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)

        for val in unique_labels:
            if val == 0: continue
            mask = (label_map == val).astype(np.uint8) * 255

            ring_color = self.get_average_color(img_rgb, mask)

            mask_bool = mask > 0
            if np.any(mask_bool):
                v_vals = hsv_image[:, :, 2][mask_bool]
                s_vals = hsv_image[:, :, 1][mask_bool]
                dark_ratio = float(np.mean(v_vals < 90))
                sat_med = float(np.median(s_vals))
                if dark_ratio > 0.45 and sat_med < 95:
                    ring_color = (0, 0, 0)

            colour_name = self._classify_allowed_color(ring_color)
            if colour_name is None:
                continue
            

            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if len(cnt) < 5: continue

                ellipse = cv2.fitEllipse(cnt)
                (center, axes, angle) = ellipse
                cx, cy = int(center[0]), int(center[1])

                inertia_ratio = min(axes) / max(axes) if max(axes) != 0 else 0
                if inertia_ratio < 0.35: continue

                if 0 <= cy < h and 0 <= cx < w:
                    obj_depths = img_depth[mask > 0]
                    valid_depths = obj_depths[np.isfinite(obj_depths)]
                    if valid_depths.size == 0: continue
                    avg_ring_depth = np.median(valid_depths)

                    pixel_width = max(axes)
                    physical_diameter = (pixel_width * avg_ring_depth) / self.fx

                    center_depth = img_depth[cy, cx]
                    is_hollow = not np.isfinite(center_depth) or (center_depth - avg_ring_depth) > 0.15

                    height = self.estimate_height_from_ground(cy, avg_ring_depth, 240)
                    if 0.08 < physical_diameter < 0.3 and is_hollow and (1.4 < height < 1.8):

                        results.append({
                            'ellipse': ellipse,
                            'color': ring_color,
                            'color_name': colour_name,
                            'depth': avg_ring_depth
                        })
        return results

    def localize(self, rings):
        if self.fx is None:
            return

        target_frame = 'map'
        camera_frame = 'oakd_rgb_camera_frame'

        for ring in rings:
            (center, axes, _) = ring['ellipse']
            cx_px, cy_px = center
            depth = ring['depth']

            X_cam_opt = (cx_px - self.cx_principal) * depth / self.fx
            Y_cam_opt = (cy_px - self.cy_principal) * depth / self.fy

            pt_cam = PointStamped()
            pt_cam.header.frame_id = camera_frame
            pt_cam.header.stamp = Time().to_msg()
            pt_cam.header.frame_id 
            pt_cam.point.x = float(depth)
            pt_cam.point.y = float(-X_cam_opt)
            pt_cam.point.z = float(-Y_cam_opt)

            try:
                pt_world = self.tf_buffer.transform(
                    pt_cam, target_frame,
                    timeout=rclpy.duration.Duration(seconds=0.1)
                )
            except Exception as e:
                self.get_logger().warn(f'TF transform failed: {e}')
                continue

            pos = np.array([pt_world.point.x, pt_world.point.y, pt_world.point.z])
            self.ring_map.update(pos, ring['color'],
                                  color_name=ring.get('color_name', 'unknown'))

        self._publish_confirmed()


    def _publish_confirmed(self):
        confirmed = self.ring_map.confirmed_landmarks()
        # Defence-in-depth: drop any landmark whose colour didn't classify
        # as one of the allowed ring colours, even if it somehow ended
        # up in the map. Rings are red / green / blue / black ONLY.
        ALLOWED = {'red', 'green', 'blue', 'black'}
        confirmed = [
            lm for lm in confirmed
            if getattr(lm, 'color_name', None) in ALLOWED
        ]
        # Drop rings detected outside the first-room AABB (second-room
        # / outside-map hits are noise; rings only live in room 1).
        FIRST_ROOM_AABB = (-4.5, 1.4, -4.5, 0.7)  # x_min, x_max, y_min, y_max
        x_min, x_max, y_min, y_max = FIRST_ROOM_AABB
        confirmed = [
            lm for lm in confirmed
            if x_min <= float(lm.position[0]) <= x_max
            and y_min <= float(lm.position[1]) <= y_max
        ]

        now = self.get_clock().now().to_msg()
        marker_array = MarkerArray()

        # On the very first publish, send a DELETEALL so any stale ring
        # markers cached in RViz from a previous node run get cleared.
        if not getattr(self, '_marker_cleared', False):
            clear = Marker()
            clear.header.frame_id = 'map'
            clear.header.stamp = now
            clear.ns = 'confirmed_rings'
            clear.action = Marker.DELETEALL
            marker_array.markers.append(clear)
            clear_labels = Marker()
            clear_labels.header.frame_id = 'map'
            clear_labels.header.stamp = now
            clear_labels.ns = 'confirmed_rings_labels'
            clear_labels.action = Marker.DELETEALL
            marker_array.markers.append(clear_labels)
            self._marker_cleared = True

        for display_n, lm in enumerate(confirmed, start=1):

            # Small coloured dot at the ring's map position.
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
            sphere.scale = Vector3(x=0.10, y=0.10, z=0.10)
            sphere.color = ColorRGBA(r=r, g=g, b=b, a=1.0)
            sphere.lifetime.sec = 0
            marker_array.markers.append(sphere)

            # Floating red label so it's easy to spot in RViz. We number
            # by position in confirmed_landmarks() (1..N dense) so
            # discarded/unconfirmed landmarks don't leave gaps in the
            # visible numbering. Marker `id` still uses lm.id to keep
            # marker identity stable across publishes.
            label = Marker()
            label.header.frame_id = 'map'
            label.header.stamp = now
            label.ns = 'confirmed_rings_labels'
            label.id = lm.id
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x = float(lm.position[0])
            label.pose.position.y = float(lm.position[1])
            label.pose.position.z = float(lm.position[2]) + 0.20
            label.pose.orientation.w = 1.0
            label.scale = Vector3(x=0.0, y=0.0, z=0.30)
            label.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
            label.text = f'ring{display_n}'
            label.lifetime.sec = 0
            marker_array.markers.append(label)

        if len(marker_array.markers) > 0:
            self.ring_pub.publish(marker_array)

    def detect_rings(self, img_rgb, img_depth):
        _, roi_mask = self.get_roi(img_rgb, img_depth, self.ground_mask_image)

        masked_rgb = np.zeros_like(img_rgb)
        masked_rgb[roi_mask] = img_rgb[roi_mask]

        from color_segmentation import ObjectDetector
        detector = ObjectDetector()
        label_map = detector.get_labels(
            masked_rgb,
            downscale_factor=1,
            n_clusters=7,
            sample_size=8_000,
            min_area=100,
            max_area=8500,
            morph_kernel_size=3,
            morph_iterations=1,
        )
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