#!/usr/bin/env python3

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.action import ActionClient
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Bool
from geometry_msgs.msg import TwistStamped, PoseStamped, Quaternion
from nav2_msgs.action import NavigateToPose

# ── Tuning parameters ────────────────────────────────────────────────────────

LINEAR_SPEED          = 0.15   # m/s forward speed while following
ANGULAR_GAIN          = 1.2    # proportional gain for heading correction
MAX_LOOKAHEAD_M       = 0.6    # furthest point to consider for lookahead (metres)
SEARCH_SPEED          = 0.4    # rad/s rotation when searching
LINE_TIMEOUT_S        = 1.0    # seconds without line before entering search mode

# Intersection tuning parameters
CHOSEN_ROUTE          = 'right'  # Options: 'left' or 'right'
INTERSECTION_THRESH   = 0.20     # Width threshold (meters) of line points to flag intersection
INTERSECTION_COOLDOWN = 3.0      # Seconds to wait before detecting another intersection
INTERSECTION_FILTER_S = 3.0      # How long to enforce lateral-preference selection after detection

START_TRANSLATION = [2.759, -0.235, 0.000]
START_ROTATION    = [0.000, 0.000, -0.701, 0.713]


class BlueLineFollower(Node):

    def __init__(self):
        super().__init__('blue_line_follower')

        self.sub = self.create_subscription(
            PointCloud2,
            '/line_detector/blue',
            self.pointcloud_callback,
            10
        )

        self.cmd_sub = self.create_subscription(
            Bool,
            '/line_follower/start_command',
            self.command_callback,
            10
        )

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.cmd_pub = self.create_publisher(TwistStamped, '/cmd_vel', qos)

        self.line_pts = None
        self.last_line_time = None

        self.last_intersection_time = None
        self.state = 'waiting'

        import tf2_ros
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.nav2_client = ActionClient(self, NavigateToPose, '/navigate_to_pose')
        self.nav2_future = None
        self.nav2_result_future = None

        self.create_timer(0.05, self.control_loop)
        self.get_logger().info(
            'Blue line follower started. Waiting for start command on /line_follower/start_command'
        )

    # ── Callbacks ────────────────────────────────────────────────────────────

    def pointcloud_callback(self, msg: PointCloud2):
        """Unpack xyz points from the blue line pointcloud."""
        if msg.width == 0:
            return

        buf = np.frombuffer(msg.data, dtype=np.float32).reshape(-1, 4)
        if len(buf) == 0:
            return

        self.line_pts = buf[:, :3].copy()
        self.last_line_time = self.get_clock().now()

    def command_callback(self, msg: Bool):
        """Handle start command to begin navigation and line following."""
        if msg.data and self.state == 'waiting':
            self.get_logger().info('🚀 Start command received! Transitioning to navigate to start pose...')
            self.state = 'navigating_to_start'

    # ── TF helper ────────────────────────────────────────────────────────────

    def _get_robot_pose(self):
        """Returns (x, y, yaw) of base_link in map frame, or None on failure."""
        try:
            tf = self.tf_buffer.lookup_transform(
                'map', 'base_link',
                Time(), timeout=Duration(seconds=0.05)
            )
        except Exception:
            return None

        x = tf.transform.translation.x
        y = tf.transform.translation.y

        q = tf.transform.rotation
        yaw = np.arctan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )
        return x, y, yaw

    # ── Main control loop ─────────────────────────────────────────────────────

    def control_loop(self):
        if self.state == 'waiting':
            # Do not publish zero velocities while waiting — avoids overriding
            # teleop or Nav2 commands on /cmd_vel before the follower is started.
            return

        elif self.state == 'navigating_to_start':
            if self._navigate_to_start():
                self.get_logger().info('✓ Start pose reached! Beginning line following...')
                self.state = 'line_following'
            return

        elif self.state != 'line_following':
            return

        # ── Line Following Logic ──────────────────────────────────────────────
        pose = self._get_robot_pose()
        if pose is None:
            return
        rx, ry, ryaw = pose

        now = self.get_clock().now()
        line_age = (
            (now - self.last_line_time).nanoseconds / 1e9
            if self.last_line_time is not None else float('inf')
        )

        # ── No line visible → search ──────────────────────────────────────────
        if self.line_pts is None or line_age > LINE_TIMEOUT_S:
            self._publish_vel(0.0, SEARCH_SPEED)
            return

        pts = self.line_pts  # (N, 3) map frame

        # ── Transform points to robot local frame ─────────────────────────────
        cos_yaw = np.cos(ryaw)
        sin_yaw = np.sin(ryaw)

        dx = pts[:, 0] - rx
        dy = pts[:, 1] - ry

        # Local X is forward, local Y is lateral left
        local_x =  dx * cos_yaw + dy * sin_yaw
        local_y = -dx * sin_yaw + dy * cos_yaw

        # ── Intersection detection ────────────────────────────────────────────
        look_zone_mask = (local_x > 0.1) & (local_x < 0.6)

        if look_zone_mask.any():
            lateral_spread = (
                np.max(local_y[look_zone_mask]) - np.min(local_y[look_zone_mask])
            )

            if lateral_spread > INTERSECTION_THRESH:
                time_since_last = (
                    (now - self.last_intersection_time).nanoseconds / 1e9
                    if self.last_intersection_time is not None else float('inf')
                )
                if time_since_last > INTERSECTION_COOLDOWN:
                    self.get_logger().info(
                        f'⚠️  INTERSECTION DETECTED! Choosing route: {CHOSEN_ROUTE.upper()}'
                    )
                    self.last_intersection_time = now

        # ── Determine whether we are actively inside a junction ───────────────
        in_junction = (
            self.last_intersection_time is not None
            and (now - self.last_intersection_time).nanoseconds / 1e9 < INTERSECTION_FILTER_S
        )

        # ── Filter points to chosen route half-plane during junction ──────────
        if in_junction:
            if CHOSEN_ROUTE == 'left':
                route_mask = local_y >= -0.05
            else:
                route_mask = local_y <= 0.05

            pts     = pts[route_mask]
            local_x = local_x[route_mask]
            local_y = local_y[route_mask]

        if len(pts) == 0:
            self._publish_vel(0.0, SEARCH_SPEED)
            return

        # ── End-of-line detection ─────────────────────────────────────────────
        ahead_mask = local_x > 0.05
        if not ahead_mask.any():
            self.get_logger().info('Reached end of line — stopping')
            self._publish_vel(0.0, 0.0)
            return

        # ── Pick lookahead point ──────────────────────────────────────────────
        #
        # Normal following: distance-based (closest point to MAX_LOOKAHEAD_M).
        #
        # During a junction: switch to lateral-preference selection — always
        # pick the rightmost (or leftmost) point within the forward cone.
        # This prevents branch-jumping in triangular roundabouts where two
        # branches can be nearly equidistant, making distance-based selection
        # unstable frame-to-frame.
        #
        cone_mask = ahead_mask & (local_x < MAX_LOOKAHEAD_M)
        if not cone_mask.any():
            # Nothing within cone — fall back to all forward points
            cone_mask = ahead_mask

        cone_indices = np.where(cone_mask)[0]

        if in_junction:
            # Lateral-preference: pick the point on the desired branch
            if CHOSEN_ROUTE == 'right':
                lookahead_idx = cone_indices[np.argmin(local_y[cone_mask])]   # most negative = rightmost
            else:
                lookahead_idx = cone_indices[np.argmax(local_y[cone_mask])]   # most positive = leftmost
        else:
            # Distance-based: closest point to MAX_LOOKAHEAD_M
            dists         = np.sqrt(local_x ** 2 + local_y ** 2)
            lookahead_idx = cone_indices[
                np.argmin(np.abs(dists[cone_mask] - MAX_LOOKAHEAD_M))
            ]

        target = pts[lookahead_idx, :2]   # (x, y) in map frame

        # ── Pure pursuit steering ─────────────────────────────────────────────
        tdx = target[0] - rx
        tdy = target[1] - ry
        angle_to_target = np.arctan2(tdy, tdx)
        heading_error   = angle_to_target - ryaw

        # Wrap to [-pi, pi]
        heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi

        angular = ANGULAR_GAIN * heading_error
        linear  = LINEAR_SPEED * max(0.0, 1.0 - abs(heading_error) / np.pi)

        self._publish_vel(linear, angular)

    # ── Velocity publisher ────────────────────────────────────────────────────

    def _publish_vel(self, linear, angular):
        msg = TwistStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.twist.linear.x  = float(linear)
        msg.twist.angular.z = float(angular)
        self.cmd_pub.publish(msg)

    # ── Nav2 helpers ──────────────────────────────────────────────────────────

    def _navigate_to_start(self):
        """Navigate to the predefined start pose using Nav2.
        Returns True when reached, False while still navigating."""

        # Phase 3: waiting for execution result
        if self.nav2_result_future is not None:
            if self.nav2_result_future.done():
                try:
                    self.nav2_result_future.result()
                    self.get_logger().info('✓ Reached start pose via Nav2')
                    self.nav2_result_future = None
                    self.nav2_future        = None
                    return True
                except Exception as e:
                    self.get_logger().error(f'Nav2 goal execution failed: {e}')
                    self.nav2_result_future = None
                    self.nav2_future        = None
                    return False
            return False

        # Phase 2: goal submitted, waiting for handle
        if self.nav2_future is not None:
            if self.nav2_future.done():
                try:
                    goal_handle = self.nav2_future.result()
                except Exception as e:
                    self.get_logger().error(f'Nav2 goal submission failed: {e}')
                    self.nav2_future = None
                    return False

                if not goal_handle.accepted:
                    self.get_logger().warn('Nav2 goal was rejected, retrying...')
                    self.nav2_future = None
                    return False

                self.nav2_result_future = goal_handle.get_result_async()
                self.nav2_future        = None
            return False

        # Phase 1: send the goal
        if not self.nav2_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().warn('Nav2 action server not available yet, retrying...')
            return False

        goal            = NavigateToPose.Goal()
        goal.pose       = PoseStamped()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp    = self.get_clock().now().to_msg()

        goal.pose.pose.position.x    = float(START_TRANSLATION[0])
        goal.pose.pose.position.y    = float(START_TRANSLATION[1])
        goal.pose.pose.position.z    = float(START_TRANSLATION[2])
        goal.pose.pose.orientation.x = float(START_ROTATION[0])
        goal.pose.pose.orientation.y = float(START_ROTATION[1])
        goal.pose.pose.orientation.z = float(START_ROTATION[2])
        goal.pose.pose.orientation.w = float(START_ROTATION[3])

        self.get_logger().info(
            f'Sending start pose goal to Nav2: ({START_TRANSLATION[0]}, {START_TRANSLATION[1]})'
        )
        self.nav2_future = self.nav2_client.send_goal_async(goal)
        return False


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(BlueLineFollower())


if __name__ == '__main__':
    main()