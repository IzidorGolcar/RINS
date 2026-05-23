#!/usr/bin/env python3

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import PointCloud2, LaserScan
from geometry_msgs.msg import TwistStamped
from visualization_msgs.msg import Marker, MarkerArray

# ── Tuning parameters ────────────────────────────────────────────────────────
LINEAR_SPEED          = 0.15   # m/s forward speed while following
ANGULAR_GAIN         = 1.4    # proportional gain for heading correction
LOOKAHEAD_M          = 0.35   # meters ahead on the line to aim for
SEARCH_SPEED         = 0.4    # rad/s rotation when searching
LINE_TIMEOUT_S       = 1.2    # seconds without line before triggering dead-end
INTERSECTION_THRESH   = 0.28   # Width threshold (meters) to identify intersection
NODE_RADIUS_MATCH    = 0.40   # Distance (meters) to consider robot at an existing node
INTERSECTION_COOLDOWN = 4.0   # Seconds between processing discrete intersections

# LiDAR Tuning Parameters
OBSTACLE_THRESHOLD_M  = 0.40   # Distance (meters) from a wall to safely stop and declare a dead end
BUMPER_BOX_WIDTH_M    = 0.15   # Narrowed to 16cm total — avoids parallel wall false positives
MIN_OBSTACLE_POINTS   = 3      # Minimum lidar points required to confirm a real obstacle


LIDAR_ANGLE_OFFSET = np.pi / 2

class RouteNode:
    """Represents a discovered intersection in the environment."""
    def __init__(self, node_id, x, y):
        self.id = node_id
        self.x = x
        self.y = y
        self.branches = {'left': None, 'right': None, 'straight': None}

class DeadEndLocation:
    """Represents a physical dead end caught by line loss or LiDAR walls."""
    def __init__(self, de_id, x, y):
        self.id = de_id
        self.x = x
        self.y = y


class AutonomousExplorer(Node):

    def __init__(self):
        super().__init__('autonomous_explorer')

        self.debug_pub = self.create_publisher(LaserScan, '/debug_bumper_scan', 10)

        # 1. Point Cloud Subscriber
        self.sub = self.create_subscription(
            PointCloud2, '/line_detector/blue', self.pointcloud_callback, 10
        )

        # 2. LiDAR Subscriber with explicit Best-Effort matching for Gazebo stability
        lidar_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )
        self.lidar_sub = self.create_subscription(
            LaserScan, '/scan', self.lidar_callback, lidar_qos
        )

        # 3. Publishers
        qos_vel = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.cmd_pub = self.create_publisher(TwistStamped, '/cmd_vel', qos_vel)
        self.marker_pub = self.create_publisher(MarkerArray, '/exploration_markers', 10)

        # Graph and Map state memory
        self.nodes = []
        self.dead_ends = []
        self.node_counter = 0
        self.dead_end_counter = 0

        self.current_node = None
        self.exploration_stack = []

        # State Machine: 'FOLLOWING', 'BACKTRACKING', 'DONE'
        self.state = 'FOLLOWING'
        self.chosen_branch = None

        # Sensor data and timing buffers
        self.line_pts = None
        self.last_line_time = None
        self.last_intersection_time = None
        self.closest_forward_obstacle = float('inf')

        # Rolling history buffer for LiDAR smoothing (avoids single-frame spikes)
        self._obstacle_history = []

        import tf2_ros
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.create_timer(0.05, self.control_loop)
        self.get_logger().info('🤖 Complete Autonomous Map Explorer Node operational.')

    def pointcloud_callback(self, msg: PointCloud2):
        if msg.width == 0:
            return
        buf = np.frombuffer(msg.data, dtype=np.float32).reshape(-1, 4)
        if len(buf) == 0:
            return
        self.line_pts = buf[:, :3].copy()
        self.last_line_time = self.get_clock().now()

    def lidar_callback(self, msg: LaserScan):
        """
        Uses explicit Cartesian coordinate projection to track objects ahead.
        Bypasses any index-rotation quirks of standard 360-degree scanners.

        Fix: Narrowed bumper box width + minimum point density requirement to
        prevent parallel walls from triggering false obstacle detections.
        Rolling minimum over last 5 frames smooths out single-frame spikes.
        """
        ranges = np.array(msg.ranges)
        angle_min = msg.angle_min
        angle_increment = msg.angle_increment

        # Calculate exact matching angle for every element
        angles = angle_min + np.arange(len(ranges)) * angle_increment
        angles = angles + LIDAR_ANGLE_OFFSET


        # Strip out readings below self-collision threshold (anything closer than 18cm)
        valid_indices = (ranges > 0.18) & (ranges < msg.range_max)
        valid_ranges = ranges[valid_indices]
        valid_angles = angles[valid_indices]

        if len(valid_ranges) == 0:
            self.closest_forward_obstacle = float('inf')
            return

        # Project points from polar to local robot frame (X=Ahead, Y=Left)
        local_x = valid_ranges * np.cos(valid_angles)
        local_y = valid_ranges * np.sin(valid_angles)

        # Tightened forward-facing detection box:
        #   - X range matches reaction distance (0.10m to 0.55m ahead)
        #   - Y range narrowed to 8cm each side to exclude parallel walls
        in_bumper_mask = (
            (local_x > 0.10) &
            (local_x < 0.55) &
            (np.abs(local_y) < BUMPER_BOX_WIDTH_M)
        )
        forward_obstacles = local_x[in_bumper_mask]

        # Require minimum point density — a single stray reflection won't stop the robot
        if len(forward_obstacles) >= MIN_OBSTACLE_POINTS:
            raw = float(np.min(forward_obstacles))
        else:
            raw = float('inf')

        # Rolling minimum over last 5 frames to smooth transient noise
        self._obstacle_history.append(raw)
        if len(self._obstacle_history) > 5:
            self._obstacle_history.pop(0)
        self.closest_forward_obstacle = min(self._obstacle_history)

        # Build a debug LaserScan containing only points inside the bumper box
        debug_msg = LaserScan()
        debug_msg.header = msg.header
        debug_msg.angle_min = msg.angle_min
        debug_msg.angle_max = msg.angle_max
        debug_msg.angle_increment = msg.angle_increment
        debug_msg.range_min = msg.range_min
        debug_msg.range_max = msg.range_max
        debug_msg.time_increment = msg.time_increment
        debug_msg.scan_time = msg.scan_time

        # Start with all ranges set to 0 (invisible in RViz), only fill in bumper box hits
        debug_ranges = np.zeros(len(ranges), dtype=np.float32)

        # Reconstruct which original indices passed the bumper box mask
        all_angles = angle_min + np.arange(len(ranges)) * angle_increment
        all_angles = all_angles + LIDAR_ANGLE_OFFSET  # ← add this
        all_local_x = ranges * np.cos(all_angles)
        all_local_y = ranges * np.sin(all_angles)
        full_bumper_mask = (
            (all_local_x > 0.10) &
            (all_local_x < 0.55) &
            (np.abs(all_local_y) < BUMPER_BOX_WIDTH_M) &
            (ranges > 0.18) &
            (ranges < msg.range_max)
        )
        debug_ranges[full_bumper_mask] = ranges[full_bumper_mask]
        debug_msg.ranges = debug_ranges.tolist()
        self.debug_pub.publish(debug_msg)

    def _get_robot_pose(self):
        try:
            tf = self.tf_buffer.lookup_transform('map', 'base_link', Time(), timeout=Duration(seconds=0.05))
            q = tf.transform.rotation
            yaw = np.arctan2(2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z))
            return tf.transform.translation.x, tf.transform.translation.y, yaw
        except Exception:
            return None

    def control_loop(self):
        pose = self._get_robot_pose()
        if pose is None:
            return
        rx, ry, ryaw = pose
        now = self.get_clock().now()

        line_age = ((now - self.last_line_time).nanoseconds / 1e9
                    if self.last_line_time is not None else float('inf'))

        if self.state == 'DONE':
            self._publish_vel(0.0, 0.0)
            return

        # ── Safe Distance Wall Checking ──────────────────────────────────────
        wall_detected = self.closest_forward_obstacle <= OBSTACLE_THRESHOLD_M
        line_lost = (self.line_pts is None or line_age > LINE_TIMEOUT_S)

        if (line_lost or wall_detected) and self.state == 'FOLLOWING':
            reason = f"Wall at {self.closest_forward_obstacle:.2f}m 🧱" if wall_detected else "Track Disappeared 🚫"
            self.get_logger().warn(f"🛑 Dead end logged! Reason: {reason}. Initiating Spin.")

            # Map out this dead end coordinate
            self.dead_end_counter += 1
            self.dead_ends.append(DeadEndLocation(self.dead_end_counter, rx, ry))
            self.publish_markers()

            self.state = 'BACKTRACKING'

        if self.state == 'BACKTRACKING':
            # Spin out of dead ends
            self._publish_vel(-0.02, SEARCH_SPEED)
            # Re-engage path finding once turned around facing the line and space clears up
            if self.line_pts is not None and line_age < 0.2 and self.closest_forward_obstacle > (OBSTACLE_THRESHOLD_M + 0.15):
                self.state = 'FOLLOWING'
            return

        # Convert line data points to Robot Local Frame
        pts = self.line_pts
        cos_yaw, sin_yaw = np.cos(ryaw), np.sin(ryaw)
        dx, dy = pts[:, 0] - rx, pts[:, 1] - ry
        local_x = dx * cos_yaw + dy * sin_yaw
        local_y = -dx * sin_yaw + dy * cos_yaw

        # ── State: FOLLOWING & Intersection Check ────────────────────────────
        if self.state == 'FOLLOWING':
            look_zone = (local_x > 0.1) & (local_x < 0.6)
            if look_zone.any():
                lateral_spread = np.max(local_y[look_zone]) - np.min(local_y[look_zone])

                time_since_inter = ((now - self.last_intersection_time).nanoseconds / 1e9
                                   if self.last_intersection_time is not None else float('inf'))

                if lateral_spread > INTERSECTION_THRESH and time_since_inter > INTERSECTION_COOLDOWN:
                    self.last_intersection_time = now
                    self._handle_intersection(rx, ry, local_x, local_y, look_zone)

        # ── Apply Active Junction Route Pruning ──────────────────────────────
        if self.chosen_branch and self.last_intersection_time is not None:
            if (now - self.last_intersection_time).nanoseconds / 1e9 < 2.0:
                if self.chosen_branch == 'left':
                    mask = local_y >= -0.05
                elif self.chosen_branch == 'right':
                    mask = local_y <= 0.05
                else:
                    mask = (local_y > -0.15) & (local_y < 0.15)

                pts = pts[mask]
                local_x = local_x[mask]
                local_y = local_y[mask]

        if len(pts) == 0:
            self._publish_vel(0.0, SEARCH_SPEED)
            return

        # ── Pure Pursuit Core ────────────────────────────────────────────────
        ahead_mask = local_x > 0.05
        if not ahead_mask.any():
            if len(self.exploration_stack) == 0:
                self.get_logger().info('🏆 Map execution finished. Stopping.')
                self.state = 'DONE'
            else:
                self.state = 'BACKTRACKING'
            return

        dists = np.sqrt(local_x**2 + local_y**2)
        ahead_indices = np.where(ahead_mask)[0]
        lookahead_idx = ahead_indices[int(np.argmin(np.abs(dists[ahead_indices] - LOOKAHEAD_M)))]

        target = pts[lookahead_idx, :2]
        heading_error = np.arctan2(target[1] - ry, target[0] - rx) - ryaw
        heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi

        angular = ANGULAR_GAIN * heading_error
        linear = LINEAR_SPEED * max(0.1, 1.0 - abs(heading_error) / np.pi)
        self._publish_vel(linear, angular)

    def _handle_intersection(self, rx, ry, local_x, local_y, look_zone):
        matched_node = None
        for n in self.nodes:
            if np.hypot(n.x - rx, n.y - ry) < NODE_RADIUS_MATCH:
                matched_node = n
                break

        if matched_node is None:
            self.node_counter += 1
            matched_node = RouteNode(self.node_counter, rx, ry)

            if np.any(local_y[look_zone] > 0.18):
                matched_node.branches['left'] = None
            if np.any(local_y[look_zone] < -0.18):
                matched_node.branches['right'] = None
            if np.any((local_y[look_zone] >= -0.12) & (local_y[look_zone] <= 0.12)):
                matched_node.branches['straight'] = None

            self.nodes.append(matched_node)
            self.get_logger().info(f"✨ Discovered Node {matched_node.id}")
        else:
            self.get_logger().info(f"📍 Returned to existing Node {matched_node.id}")

        self.current_node = matched_node

        chosen = None
        for direction in ['left', 'right', 'straight']:
            if self.current_node.branches[direction] is None:
                chosen = direction
                break

        if chosen:
            self.current_node.branches[chosen] = True
            if self.current_node not in self.exploration_stack:
                self.exploration_stack.append(self.current_node)

            self.chosen_branch = chosen
            self.state = 'FOLLOWING'
            self.get_logger().info(f"🚀 Exploring branch [{chosen.upper()}] from Node {self.current_node.id}")
        else:
            if self.exploration_stack:
                if self.current_node in self.exploration_stack:
                    self.exploration_stack.remove(self.current_node)

                if len(self.exploration_stack) > 0:
                    self.state = 'BACKTRACKING'
                else:
                    self.state = 'DONE'
            else:
                self.state = 'DONE'

        self.publish_markers()

    def publish_markers(self):
        """Generates visual elements across the graph network mapping structures."""
        marker_array = MarkerArray()
        time_now = self.get_clock().now().to_msg()

        # 1. INTERSECTIONS (Cyan Spheres)
        for node in self.nodes:
            sphere = Marker()
            sphere.header.frame_id = "map"
            sphere.header.stamp = time_now
            sphere.ns = "intersections"
            sphere.id = node.id
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position.x, sphere.pose.position.y = float(node.x), float(node.y)
            sphere.pose.position.z = 0.05
            sphere.scale.x, sphere.scale.y, sphere.scale.z = 0.25, 0.25, 0.25
            sphere.color.r, sphere.color.g, sphere.color.b, sphere.color.a = 0.0, 0.9, 0.9, 0.85
            marker_array.markers.append(sphere)

            # Floating text labels
            text = Marker()
            text.header.frame_id = "map"
            text.header.stamp = time_now
            text.ns = "node_labels"
            text.id = node.id + 1000
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x, text.pose.position.y, text.pose.position.z = float(node.x), float(node.y), 0.35
            text.scale.z = 0.12
            text.color.r, text.color.g, text.color.b, text.color.a = 1.0, 1.0, 1.0, 1.0
            active_branches = [k for k, v in node.branches.items() if v is not None]
            text.text = f"Node {node.id}\nPaths: {active_branches}"
            marker_array.markers.append(text)

        # 2. DEAD ENDS (Crimson Red Cubes)
        for de in self.dead_ends:
            cube = Marker()
            cube.header.frame_id = "map"
            cube.header.stamp = time_now
            cube.ns = "dead_ends"
            cube.id = de.id
            cube.type = Marker.CUBE
            cube.action = Marker.ADD
            cube.pose.position.x, cube.pose.position.y = float(de.x), float(de.y)
            cube.pose.position.z = 0.05
            cube.scale.x, cube.scale.y, cube.scale.z = 0.22, 0.22, 0.22
            cube.color.r, cube.color.g, cube.color.b, cube.color.a = 0.9, 0.1, 0.1, 0.90
            marker_array.markers.append(cube)

        if marker_array.markers:
            self.marker_pub.publish(marker_array)

    def _publish_vel(self, linear, angular):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.twist.linear.x = float(linear)
        msg.twist.angular.z = float(angular)
        self.cmd_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(AutonomousExplorer())

if __name__ == '__main__':
    main()