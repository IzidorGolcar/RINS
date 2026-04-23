#!/usr/bin/python3

import rclpy
from rclpy.node import Node
import cv2, math
import numpy as np
import tf2_ros

from sensor_msgs.msg import Image, CompressedImage
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


class RingDetector(Node):

    MARKER_QOS = QoSProfile(
        durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        reliability=QoSReliabilityPolicy.RELIABLE,
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=10)

    def __init__(self):
        super().__init__('transform_point')

        self.bridge = CvBridge()

        self.rgb_sub = message_filters.Subscriber(self, CompressedImage, '/gemini/color/image_raw/compressed')
        self.depth_sub = message_filters.Subscriber(self, CompressedImage, '/gemini/depth/image_raw/compressedDepth')

        self.stream = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=20,
            slop=0.5,
            allow_headerless=True
        )

        self.stream.registerCallback(self.stream_callback)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.ring_pub = self.create_publisher(MarkerArray, '/ring_markers', self.MARKER_QOS)
        self.marker_id = 0

        self.received_camera_info = False
        self.fx = self.fy = None
        self.cx_principal = self.cy_principal = None
        
        
        camera_info_topic = '/gemini/color/camera_info'
        self.create_subscription(CameraInfo, camera_info_topic,
                                 self.cam_info_callback, 10)
        self.create_subscription(CameraInfo, camera_info_topic,
                                 self.cam_info_callback, qos_profile_sensor_data)

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


    def stream_callback(self, rgb_msg, depth_msg):
        if not self.received_camera_info:
            return
        try:
            rgb_payload = np.frombuffer(rgb_msg.data, dtype=np.uint8)
            cv_image = cv2.imdecode(rgb_payload, cv2.IMREAD_COLOR)

            depth_fmt, depth_header = depth_msg.format, depth_msg.data[:12]
            depth_payload = np.frombuffer(depth_msg.data[12:], dtype=np.uint8)
            img_depth_raw = cv2.imdecode(depth_payload, cv2.IMREAD_UNCHANGED)

            # Convert 16-bit mm to float32 meters for easier math
            # This makes 'depth' values like 1.5 instead of 1500
            img_depth_meters = img_depth_raw.astype(np.float32) / 1000.0

            self.detect_rings(cv_image.copy(), img_depth_meters.copy(), rgb_msg.header.stamp)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f'{e}')
        

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

    def find_rings(self, label_map, img_rgb, img_depth):
        results = []
        (h, w) = label_map.shape
        unique_labels = np.unique(label_map)

        for val in unique_labels:
            if val == 0: continue
            mask = (label_map == val).astype(np.uint8) * 255

            ring_color = self.get_average_color(img_rgb, mask)            

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
                    if 0.08 < physical_diameter < 0.3 and is_hollow and (0.3 < height < 2.0):
                        results.append({
                            'ellipse': ellipse,
                            'color': ring_color,
                            'depth': avg_ring_depth
                        })
        return results

    def _project_to_map(self, cx, cy, z, stamp):
        if self.fx is None or self.cx_principal is None:
            return None
        z_opt = float(z)
        x_opt = (cx - self.cx_principal) * z_opt / self.fx
        y_opt = (cy - self.cy_principal) * z_opt / self.fy

        pt = PointStamped()
        pt.header.frame_id = 'gemini_color_optical_frame' 
        pt.header.stamp = stamp
        pt.point.x = x_opt
        pt.point.y = y_opt
        pt.point.z = z_opt

        stamp_time = Time.from_msg(stamp)
        if self.tf_buffer.can_transform('map', pt.header.frame_id, stamp_time, 
                                        timeout=rclpy.duration.Duration(seconds=0.05)):
            try:
                pt_map = self.tf_buffer.transform(pt, 'map')
                return np.array([pt_map.point.x, pt_map.point.y, pt_map.point.z])
            except:
                pass

        pt.header.stamp = Time().to_msg()
        try:
            pt_map = self.tf_buffer.transform(pt, 'map', timeout=rclpy.duration.Duration(seconds=0.05))
            return np.array([pt_map.point.x, pt_map.point.y, pt_map.point.z])
        except Exception as e:
            self.get_logger().warn(f'TF lookup failed: {e}', throttle_duration_sec=5.0)
            return None

    def localize_rings(self, rings, stamp):
        """Call this at the end of detect_rings"""
        for ring in rings:
            (center, axes, angle) = ring['ellipse']
            cx_px, cy_px = center
            depth = ring['depth']

            pos_map = self._project_to_map(cx_px, cy_px, depth, stamp)
            
            if pos_map is not None:
                self.ring_map.update(pos_map, ring['color'])

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
            sphere.pose.position.x = float(lm.position[0])
            sphere.pose.position.y = float(lm.position[1])
            sphere.pose.position.z = float(lm.position[2])
            label.pose.orientation.w = 1.0
            sphere.scale = Vector3(x=0.15, y=0.15, z=0.15)
            label.color = ColorRGBA(r=r, g=g, b=b, a=1.0)
            label.text = f'Ring {lm.id}'
            marker_array.markers.append(label)

        if len(marker_array.markers) > 0:
            self.ring_pub.publish(marker_array)

    def detect_rings(self, img_rgb, img_depth, stamp):
        _, roi_mask = self.get_roi(img_rgb, img_depth)
        
        masked_rgb = np.zeros_like(img_rgb)
        masked_rgb[roi_mask] = img_rgb[roi_mask]
        cv2.imshow('ROI Masked RGB', masked_rgb)

        from color_segmentation import ObjectDetector
        detector = ObjectDetector()
        label_map = detector.get_labels(
            masked_rgb,
            downscale_factor=2,
            n_clusters=10,
            sample_size=10_000,
            min_area=3200,
            morph_kernel_size=7,
            morph_iterations=2,
        )
        self.display_label_map(label_map)
        
        rings = self.find_rings(label_map, img_rgb, img_depth)
        self.display_detections(img_rgb, rings)

        self.localize_rings(rings, stamp)

        


def main():
    rclpy.init(args=None)
    rd_node = RingDetector()
    rclpy.spin(rd_node)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()