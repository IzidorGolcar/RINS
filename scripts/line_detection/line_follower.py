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

LINEAR_SPEED        = 0.15   # m/s forward speed while following
ANGULAR_GAIN       = 1.2    # proportional gain for heading correction
LOOKAHEAD_M        = 0.4    # how far ahead on the line to aim for (metres)
SEARCH_SPEED       = 0.4    # rad/s rotation when searching
LINE_TIMEOUT_S     = 1.0    # seconds without line before entering search mode

# Intersection tuning parameters
CHOSEN_ROUTE       = 'left' # Options: 'left' or 'right'
INTERSECTION_THRESH = 0.25   # Width threshold (meters) of line points to flag intersection
INTERSECTION_COOLDOWN = 3.0 # Seconds to wait before detecting another intersection


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
        
        # Track last intersection time to prevent log spamming
        self.last_intersection_time = None

        # Robot pose in map frame — we get this from odom via TF
        import tf2_ros
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Control loop at 20 Hz
        self.create_timer(0.05, self.control_loop)
        self.get_logger().info(f'Blue line follower started. Router configured to turn: {CHOSEN_ROUTE}')


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

        # ── Transform points to Robot Local Frame ───────────────────────────
        # This makes intersection analysis significantly cleaner than managing it in map frame
        cos_yaw = np.cos(ryaw)
        sin_yaw = np.sin(ryaw)
        
        dx = pts[:, 0] - rx
        dy = pts[:, 1] - ry
        
        # Local X is forward, Local Y is lateral left
        local_x = dx * cos_yaw + dy * sin_yaw
        local_y = -dx * sin_yaw + dy * cos_yaw

        # ── Intersection Detection ───────────────────────────────────────────
        # Look at line points in a window slightly ahead of the robot
        look_zone_mask = (local_x > 0.1) & (local_x < 0.6)
        
        if look_zone_mask.any():
            lateral_spread = np.max(local_y[look_zone_mask]) - np.min(local_y[look_zone_mask])
            
            if lateral_spread > INTERSECTION_THRESH:
                # Check cooldown to avoid multi-triggering on the same intersection
                time_since_last = (
                    (now - self.last_intersection_time).nanoseconds / 1e9
                    if self.last_intersection_time is not None else float('inf')
                )
                if time_since_last > INTERSECTION_COOLDOWN:
                    self.get_logger().info(f'⚠️ INTERSECTION DETECTED! Choosing route: {CHOSEN_ROUTE.upper()}')
                    self.last_intersection_time = now

        # ── Filter points based on chosen route ─────────────────────────────
        # If we are inside an active intersection area, prune points from the unselected side
        # so pure pursuit targets our intended lane.
        if self.last_intersection_time is not None and (now - self.last_intersection_time).nanoseconds / 1e9 < 1.5:
            if CHOSEN_ROUTE == 'left':
                route_mask = local_y >= -0.05  # Keep left points + small buffer
            else:
                route_mask = local_y <= 0.05   # Keep right points + small buffer
                
            # Apply filter
            pts = pts[route_mask]
            local_x = local_x[route_mask]
            local_y = local_y[route_mask]

        if len(pts) == 0:
            self._publish_vel(0.0, SEARCH_SPEED)
            return

        # ── End-of-line detection ────────────────────────────────────────────
        ahead_mask = local_x > 0.05
        if not ahead_mask.any():
            self.get_logger().info('Reached end of line — stopping')
            self._publish_vel(0.0, 0.0)
            return

        # ── Pick lookahead point ─────────────────────────────────────────────
        dists = np.sqrt(local_x**2 + local_y**2)
        ahead_indices = np.where(ahead_mask)[0]
        ahead_dists = dists[ahead_indices]
        
        lookahead_idx = ahead_indices[int(np.argmin(np.abs(ahead_dists - LOOKAHEAD_M)))]
        target = pts[lookahead_idx, :2]   # (x, y) in map frame

        # ── Pure pursuit steering ────────────────────────────────────────────
        tdx = target[0] - rx
        tdy = target[1] - ry
        angle_to_target = np.arctan2(tdy, tdx)
        heading_error = angle_to_target - ryaw

        # Wrap to [-pi, pi]
        heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi

        angular = ANGULAR_GAIN * heading_error
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