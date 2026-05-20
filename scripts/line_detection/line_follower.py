#!/usr/bin/env python3

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import TwistStamped

# ── Tuning parameters ────────────────────────────────────────────────────────

LINEAR_SPEED       = 0.15   # m/s forward speed while following
ANGULAR_GAIN       = 1.2    # proportional gain for heading correction
LOOKAHEAD_M        = 0.4    # how far ahead on the line to aim for (metres)
SEARCH_SPEED       = 0.4    # rad/s rotation when searching
LINE_TIMEOUT_S     = 1.0    # seconds without line before entering search mode


class BlueLineFollower(Node):

    def __init__(self):
        super().__init__('blue_line_follower')

        self.sub = self.create_subscription(
            PointCloud2,
            '/line_detector/blue',
            self.pointcloud_callback,
            10
        )

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.cmd_pub = self.create_publisher(TwistStamped, '/cmd_vel', qos)

        # Latest line points in map frame (N,3)
        self.line_pts = None
        self.last_line_time = None

        # Robot pose in map frame — we get this from odom via TF
        import tf2_ros
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Control loop at 20 Hz
        self.create_timer(0.05, self.control_loop)
        self.get_logger().info('Blue line follower started')


    def pointcloud_callback(self, msg: PointCloud2):
        """Unpack xyz points from the class_3 pointcloud."""
        if msg.width == 0:
            return

        buf = np.frombuffer(msg.data, dtype=np.float32).reshape(-1, 4)
        if len(buf) == 0:
            return

        self.line_pts = buf[:, :3].copy()   # (N,3) in map frame
        self.last_line_time = self.get_clock().now()


    def _get_robot_pose(self):
        """
        Returns (x, y, yaw) of base_link in map frame, or None on failure.
        """
        try:
            tf = self.tf_buffer.lookup_transform(
                'map', 'base_link',
                Time(), timeout=Duration(seconds=0.05)
            )
        except Exception:
            return None

        x = tf.transform.translation.x
        y = tf.transform.translation.y

        # Quaternion → yaw
        q = tf.transform.rotation
        yaw = np.arctan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )
        return x, y, yaw


    def control_loop(self):
        pose = self._get_robot_pose()
        if pose is None:
            return
        rx, ry, ryaw = pose

        now = self.get_clock().now()
        line_age = (
            (now - self.last_line_time).nanoseconds / 1e9
            if self.last_line_time is not None else float('inf')
        )

        # ── No line visible → search ─────────────────────────────────────────
        if self.line_pts is None or line_age > LINE_TIMEOUT_S:
            self._publish_vel(0.0, SEARCH_SPEED)
            return

        pts = self.line_pts  # (N,3) map frame

        # ── Find nearest point on the line to the robot ──────────────────────
        dists = np.sqrt((pts[:, 0] - rx)**2 + (pts[:, 1] - ry)**2)
        nearest_idx = int(np.argmin(dists))
        nearest_dist = dists[nearest_idx]

        # ── End-of-line detection ────────────────────────────────────────────
        # Sort all points by distance; if the nearest point is at the far end
        # of the line (high index when sorted along path) we've reached the end.
        # Simpler heuristic: if the robot has passed all line points (every
        # point is behind the robot), stop.
        # Project each point onto the robot's forward axis
        forward = np.array([np.cos(ryaw), np.sin(ryaw)])
        rel = pts[:, :2] - np.array([rx, ry])
        projections = rel @ forward   # positive = ahead, negative = behind

        ahead_mask = projections > 0.05
        if not ahead_mask.any():
            # All line points are behind the robot — end of line
            self.get_logger().info('Reached end of line — stopping')
            self._publish_vel(0.0, 0.0)
            return

        # ── Pick lookahead point ─────────────────────────────────────────────
        # Among points ahead of the robot, find the one closest to LOOKAHEAD_M
        ahead_indices = np.where(ahead_mask)[0]
        ahead_dists = dists[ahead_indices]
        # target the point whose distance is closest to LOOKAHEAD_M
        lookahead_idx = ahead_indices[int(np.argmin(np.abs(ahead_dists - LOOKAHEAD_M)))]
        target = pts[lookahead_idx, :2]   # (x, y) in map frame

        # ── Pure pursuit steering ────────────────────────────────────────────
        # Angle from robot heading to target point
        dx = target[0] - rx
        dy = target[1] - ry
        angle_to_target = np.arctan2(dy, dx)
        heading_error = angle_to_target - ryaw

        # Wrap to [-pi, pi]
        heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi

        angular = ANGULAR_GAIN * heading_error
        # Slow down on sharp turns
        linear = LINEAR_SPEED * max(0.0, 1.0 - abs(heading_error) / np.pi)

        self._publish_vel(linear, angular)


    def _publish_vel(self, linear, angular):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.twist.linear.x = float(linear)
        msg.twist.angular.z = float(angular)
        self.cmd_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(BlueLineFollower())


if __name__ == '__main__':
    main()