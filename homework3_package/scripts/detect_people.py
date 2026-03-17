#!/usr/bin/env python3

import json
import math
import os

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2 as pc2

from visualization_msgs.msg import Marker

from cv_bridge import CvBridge, CvBridgeError
import cv2

from geometry_msgs.msg import PointStamped
from tf2_ros import Buffer, TransformListener, TransformException
from tf2_geometry_msgs import do_transform_point

from ultralytics import YOLO

# from rclpy.parameter import Parameter
# from rcl_interfaces.msg import SetParametersResult

class detect_faces(Node):

	def __init__(self):
		super().__init__('detect_faces')

		self.declare_parameters(
			namespace='',
			parameters=[
				('device', ''),
				('target_frame', 'map'),
				('faces_file', os.path.expanduser('~/.ros/detected_faces.json')),
				('merge_distance', 0.6),
		])

		marker_topic = "/people_marker"

		self.detection_color = (0,0,255)
		self.device = self.get_parameter('device').get_parameter_value().string_value
		self.target_frame = self.get_parameter('target_frame').get_parameter_value().string_value
		self.faces_file = os.path.expanduser(self.get_parameter('faces_file').get_parameter_value().string_value)
		self.merge_distance = self.get_parameter('merge_distance').get_parameter_value().double_value

		self.bridge = CvBridge()
		self.scan = None
		self.tf_buffer = Buffer()
		self.tf_listener = TransformListener(self.tf_buffer, self)

		self.rgb_image_sub = self.create_subscription(Image, "/oakd/rgb/preview/image_raw", self.rgb_callback, qos_profile_sensor_data)
		self.pointcloud_sub = self.create_subscription(PointCloud2, "/oakd/rgb/preview/depth/points", self.pointcloud_callback, qos_profile_sensor_data)

		self.marker_pub = self.create_publisher(Marker, marker_topic, 10)
		self.face_pub = self.create_publisher(PointStamped, '/detected_faces', 10)

		self.model = YOLO("yolo26n.pt")

		self.faces = []
		self.saved_faces = []
		self._load_saved_faces()

		# Republish previously saved faces once so robot_commander can receive them on startup
		if self.saved_faces:
			self.create_timer(2.0, self._republish_saved_faces)

		self.get_logger().info(f"Node has been initialized! Will publish face markers to {marker_topic}.")

	def _load_saved_faces(self):
		if not os.path.exists(self.faces_file):
			return
		try:
			with open(self.faces_file, 'r', encoding='utf-8') as f:
				data = json.load(f)
			for entry in data:
				self.saved_faces.append((float(entry['x']), float(entry['y']), float(entry.get('z', 0.0))))
		except Exception as e:
			self.get_logger().warn(f"Failed to load faces file: {e}")

	def _save_faces(self):
		folder = os.path.dirname(self.faces_file)
		if folder:
			os.makedirs(folder, exist_ok=True)
		payload = [{'x': x, 'y': y, 'z': z} for x, y, z in self.saved_faces]
		with open(self.faces_file, 'w', encoding='utf-8') as f:
			json.dump(payload, f, indent=2)

	def _republish_saved_faces(self):
		for fx, fy, fz in self.saved_faces:
			msg = PointStamped()
			msg.header.frame_id = self.target_frame
			msg.header.stamp = self.get_clock().now().to_msg()
			msg.point.x = fx
			msg.point.y = fy
			msg.point.z = fz
			self.face_pub.publish(msg)
		self.get_logger().info(f"Republished {len(self.saved_faces)} saved face(s) on startup.")

	def _is_new_face(self, x, y):
		for fx, fy, _ in self.saved_faces:
			if math.hypot(x - fx, y - fy) < self.merge_distance:
				return False
		return True

	def rgb_callback(self, data):

		self.faces = []

		try:
			cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")


			# run inference
			res = self.model.predict(cv_image, imgsz=(256, 320), show=False, verbose=False, classes=[0], device=self.device)

			# iterate over resultswwwwwww
			
			for x in res:
				if not x.boxes.conf.tolist():
					continue
				confidence = x.boxes.conf[0]
				bbox = x.boxes.xyxy
				bbox = bbox[0]
				if confidence < 0.4 and (bbox[0] - bbox[2])**2 < 500:
					self.get_logger().info(str(confidence))
					continue
				
				if bbox.nelement() == 0: # skip if empty
					continue

				self.get_logger().info(f"Person has been detected!")

				# draw rectangle
				cv_image = cv2.rectangle(cv_image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), self.detection_color, 3)

				cx = int((bbox[0]+bbox[2])/2)
				cy = int((bbox[1]+bbox[3])/2)
				self.get_logger().info(f"{(bbox[0] - bbox[2])**2}")

				# draw the center of bounding box
				cv_image = cv2.circle(cv_image, (cx,cy), 5, self.detection_color, -1)

				self.faces.append((cx,cy))

			cv2.imshow("image", cv_image)
			key = cv2.waitKey(1)
			if key==27:
				print("exiting")
				exit()
			
		except CvBridgeError as e:
			print(e)

	def pointcloud_callback(self, data):

		if not self.faces:
			return

		# get point cloud attributes
		height = data.height
		width = data.width

		a = pc2.read_points_numpy(data, field_names=("x", "y", "z"))
		a = a.reshape((height, width, 3))

		# iterate over face coordinates
		for x,y in self.faces:
			if x < 0 or y < 0 or x >= width or y >= height:
				continue

			# read center coordinates
			d = a[y,x,:]
			if any(math.isnan(float(v)) or math.isinf(float(v)) for v in d):
				continue

			raw_point = PointStamped()
			raw_point.header = data.header
			raw_point.point.x = float(d[0])
			raw_point.point.y = float(d[1])
			raw_point.point.z = float(d[2])

			try:
				transform = self.tf_buffer.lookup_transform(self.target_frame, data.header.frame_id, rclpy.time.Time())
				face_msg = do_transform_point(raw_point, transform)
			except TransformException:
				continue

			# create marker
			marker = Marker()

			marker.header.frame_id = self.target_frame
			marker.header.stamp = data.header.stamp

			marker.type = 2
			marker.id = len(self.saved_faces)

			# Set the scale of the marker
			scale = 0.5
			marker.scale.x = scale
			marker.scale.y = scale
			marker.scale.z = scale

			# Set the color
			marker.color.r = 0.0
			marker.color.g = 0.0
			marker.color.b = 1.0
			marker.color.a = 1.0

			# Set the pose of the marker
			marker.pose.position.x = float(face_msg.point.x)
			marker.pose.position.y = float(face_msg.point.y)
			marker.pose.position.z = float(face_msg.point.z)
			marker.pose.orientation.w = 1.0
			if 0.15 > face_msg.point.z > 0.25:
				continue 

			self.marker_pub.publish(marker)
			self.face_pub.publish(face_msg)
			self.get_logger().info(f"z: {face_msg.point.z:.2f}")
			if self._is_new_face(face_msg.point.x, face_msg.point.y):
				self.saved_faces.append((face_msg.point.x, face_msg.point.y, ace_msg.point.z))
				self._save_faces()
				self.get_logger().info(f"Saved face #{len(self.saved_faces)} at ({face_msg.point.x:.2f}, {face_msg.point.y:.2f})")

def main():
	print('Face detection node starting.')

	rclpy.init(args=None)
	node = detect_faces()
	rclpy.spin(node)
	node.destroy_node()
	rclpy.shutdown()

if __name__ == '__main__':
	main()