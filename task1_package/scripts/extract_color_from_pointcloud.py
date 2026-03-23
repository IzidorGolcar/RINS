#!/usr/bin/env python3

import math

import cv2
import numpy as np
import rclpy
import rclpy.time
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from std_msgs.msg import String
from tf2_geometry_msgs import do_transform_point
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker


class DetectRings(Node):

	def __init__(self):
		super().__init__('detect_rings')

		self.declare_parameters(
			namespace='',
			parameters=[
				('target_frame', 'map'),
				('merge_distance', 0.7),
				('ring_vote_threshold', 4),
			])

		self.target_frame = self.get_parameter('target_frame').get_parameter_value().string_value
		self.merge_distance = self.get_parameter('merge_distance').get_parameter_value().double_value
		self.ring_vote_threshold = int(self.get_parameter('ring_vote_threshold').get_parameter_value().integer_value)

		self.grid_resolution = 10.0
		self.meter_range = 10.0
		grid_size = int(self.meter_range * 2 * self.grid_resolution)
		self.ring_position_votes = {
			'red': np.zeros((grid_size, grid_size), dtype=np.float32),
			'green': np.zeros((grid_size, grid_size), dtype=np.float32),
			'blue': np.zeros((grid_size, grid_size), dtype=np.float32),
			'yellow': np.zeros((grid_size, grid_size), dtype=np.float32),
		}

		self.detected_rings = []

		self.tf_buffer = Buffer()
		self.tf_listener = TransformListener(self.tf_buffer, self)

		self.pointcloud_sub = self.create_subscription(
			PointCloud2,
			'/oakd/rgb/preview/depth/points',
			self.pointcloud_callback,
			qos_profile_sensor_data,
		)

		marker_qos = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT)
		self.marker_pub = self.create_publisher(Marker, '/ring_marker', marker_qos)
		self.ring_pub = self.create_publisher(PointStamped, '/detected_rings', 10)
		self.ring_color_pub = self.create_publisher(String, '/detected_rings_color', 10)

		self.create_timer(2.0, self.publish_ring_locations)

		self.get_logger().info('Ring detector initialized.')

	def _rgb_float_to_bgr(self, rgb_float):
		rgb_uint32 = rgb_float.view(np.uint32)
		r = ((rgb_uint32 >> 16) & 0xFF).astype(np.uint8)
		g = ((rgb_uint32 >> 8) & 0xFF).astype(np.uint8)
		b = (rgb_uint32 & 0xFF).astype(np.uint8)
		return np.dstack((b, g, r))

	def _find_ring_candidates(self, bgr_image):
		hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

		color_masks = {
			'red': cv2.inRange(hsv, (0, 90, 60), (10, 255, 255)) | cv2.inRange(hsv, (170, 90, 60), (180, 255, 255)),
			'green': cv2.inRange(hsv, (40, 70, 50), (85, 255, 255)),
			'blue': cv2.inRange(hsv, (90, 90, 40), (135, 255, 255)),
			'yellow': cv2.inRange(hsv, (18, 90, 60), (35, 255, 255)),
		}

		candidates = []
		kernel = np.ones((3, 3), np.uint8)
		for color_name, mask in color_masks.items():
			cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
			cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

			contours, hierarchy = cv2.findContours(cleaned, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
			if hierarchy is None:
				continue

			for idx, contour in enumerate(contours):
				area = cv2.contourArea(contour)
				if area < 80.0:
					continue

				perimeter = cv2.arcLength(contour, True)
				if perimeter <= 1e-6:
					continue

				circularity = 4.0 * math.pi * area / (perimeter * perimeter)
				if circularity < 0.45:
					continue

				has_hole = hierarchy[0][idx][2] != -1
				if not has_hole:
					continue

				(cx, cy), radius = cv2.minEnclosingCircle(contour)
				if radius < 8.0 or radius > 120.0:
					continue

				candidates.append((int(cx), int(cy), color_name))

		return candidates

	def _publish_ring_point(self, x, y, z, color):
		point_msg = PointStamped()
		point_msg.header.frame_id = self.target_frame
		point_msg.header.stamp = self.get_clock().now().to_msg()
		point_msg.point.x = float(x)
		point_msg.point.y = float(y)
		point_msg.point.z = float(z)
		self.ring_pub.publish(point_msg)

		color_msg = String()
		color_msg.data = f'{color}:{x:.3f},{y:.3f},{z:.3f}'
		self.ring_color_pub.publish(color_msg)

	def _is_new_ring(self, x, y, color):
		for ring in self.detected_rings:
			if ring['color'] == color and math.hypot(x - ring['x'], y - ring['y']) < self.merge_distance:
				return False
		return True

	def _add_vote(self, color, x, y):
		idx_x = int((x + self.meter_range) * self.grid_resolution)
		idx_y = int((y + self.meter_range) * self.grid_resolution)
		votes = self.ring_position_votes[color]
		if 0 <= idx_x < votes.shape[0] and 0 <= idx_y < votes.shape[1]:
			votes[idx_x, idx_y] += 1.0

	def pointcloud_callback(self, data):
		try:
			cloud = pc2.read_points_numpy(data, field_names=('x', 'y', 'z', 'rgb')).reshape((data.height, data.width, 4))
		except Exception:
			return

		rgb = cloud[:, :, 3]
		bgr = self._rgb_float_to_bgr(rgb)
		candidates = self._find_ring_candidates(bgr)
		if not candidates:
			return

		try:
			transform = self.tf_buffer.lookup_transform(self.target_frame, data.header.frame_id, rclpy.time.Time())
		except TransformException:
			return

		for u, v, color in candidates:
			if not (0 <= u < data.width and 0 <= v < data.height):
				continue

			xyz = cloud[v, u, :3]
			if np.isnan(xyz).any() or float(xyz[2]) <= 0.05 or float(xyz[2]) > 4.0:
				continue

			raw_pt = PointStamped()
			raw_pt.header = data.header
			raw_pt.point.x = float(xyz[0])
			raw_pt.point.y = float(xyz[1])
			raw_pt.point.z = float(xyz[2])

			try:
				ring_msg = do_transform_point(raw_pt, transform)
			except TransformException:
				continue

			self._add_vote(color, ring_msg.point.x, ring_msg.point.y)

			marker = Marker()
			marker.header.frame_id = self.target_frame
			marker.header.stamp = data.header.stamp
			marker.type = Marker.SPHERE
			marker.id = abs(hash((u, v, color))) % 100000
			marker.scale.x = marker.scale.y = marker.scale.z = 0.12
			marker.pose.position = ring_msg.point
			marker.pose.orientation.w = 1.0
			marker.color.a = 1.0
			if color == 'red':
				marker.color.r = 1.0
			elif color == 'green':
				marker.color.g = 1.0
			elif color == 'blue':
				marker.color.b = 1.0
			elif color == 'yellow':
				marker.color.r = 1.0
				marker.color.g = 1.0
			self.marker_pub.publish(marker)

	def publish_ring_locations(self):
		for color, votes in self.ring_position_votes.items():
			peak_indices = np.argwhere(votes >= self.ring_vote_threshold)
			for idx_x, idx_y in peak_indices:
				real_x = (idx_x / self.grid_resolution) - self.meter_range
				real_y = (idx_y / self.grid_resolution) - self.meter_range
				real_z = 0.5

				self._publish_ring_point(real_x, real_y, real_z, color)

				if self._is_new_ring(real_x, real_y, color):
					self.detected_rings.append({
						'x': real_x,
						'y': real_y,
						'z': real_z,
						'color': color,
					})
					self.get_logger().info(
						f"Detected ring #{len(self.detected_rings)} ({color}) at ({real_x:.2f}, {real_y:.2f})"
					)

			votes.fill(0)


def main():
	rclpy.init(args=None)
	node = DetectRings()
	try:
		rclpy.spin(node)
	except KeyboardInterrupt:
		pass
	finally:
		node.destroy_node()
		rclpy.shutdown()


if __name__ == '__main__':
	main()