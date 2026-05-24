#!/usr/bin/env python3
"""
Autonomous Explorer Node

States:
  FOLLOWING       -- driving along the detected line
  TURNING         -- spinning by a fixed angle (180 deg for dead-end/obstacle,
                     90 deg left for T junction)

Junction handling
  T junction  Robot arrives at stem end; points form a wide lateral bar with
              almost no forward extent.  Detected once, latched, then handled
              by a 90-deg left turn -- same reliable mechanism used for 180s.
              The cache retains the bar points so after the turn the robot
              immediately sees the bar as a forward line and resumes following.

  Y junction  Two branches diverge ahead.  Detected by a lateral gap in a
              probe band.  The leftmost branch (highest local-y) is isolated
              and a normal polynomial target is computed from it.

Camera notes
  30 cm blind spot: rolling global-frame cache keeps points alive.
  Localisation jitter: cache averages many frames, capped at CACHE_MAX_PTS.
  Spinning fix: SPIN_DEADBAND_RAD wide so gentle curves keep linear speed.
"""

import numpy as np
import rclpy
from collections import deque
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.action import ActionClient

from sensor_msgs.msg import PointCloud2, LaserScan
from geometry_msgs.msg import TwistStamped, Point, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from nav2_msgs.action import NavigateToPose
import tf2_ros

# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------
LINEAR_SPEED          = 0.15
ANGULAR_GAIN          = 1.4
LOOKAHEAD_M           = 0.45

LINE_CACHE_S          = 1.5
CACHE_MAX_PTS         = 400
FRESH_WINDOW_S        = 0.5
LINE_LOST_TIMEOUT_S   = 4.0

SPIN_DEADBAND_RAD     = 0.9
MIN_LINEAR_FRAC       = 0.25

OBSTACLE_DIST_M       = 0.40
OBSTACLE_BOX_W_M      = 0.15
MIN_OBSTACLE_PTS      = 3
LIDAR_ANGLE_OFFSET    = np.pi / 2

TURN_SPEED_RAD_S      = 0.8
TURN_TOL_RAD          = 0.08

BREADCRUMB_SPACING_M  = 0.10
FRONTIER_RADIUS_M     = 0.35
POI_MIN_SPACING_M     = 0.20

# Y-junction detection (two branches diverging ahead)
Y_PROBE_X_M           = 0.35   # forward distance of probe band
Y_PROBE_W_M           = 0.15   # half-width of probe band
Y_GAP_M               = 0.12   # lateral gap between cluster centres
Y_MIN_PTS             = 5      # min points per cluster

# T-junction detection (bar perpendicular, nothing straight ahead)
# Triggered when the point cloud has little forward depth but wide lateral spread.
T_FORWARD_DEPTH_M     = 0.20   # max forward extent to consider it a T
T_BAR_MIN_SPREAD_M    = 0.25   # min lateral spread to confirm there is a bar
T_CONFIRM_FRAMES      = 3      # must be detected this many consecutive frames
                               # before committing (prevents jitter false-fires)
# ---------------------------------------------------------------------------


class Explorer(Node):
    def __init__(self):
        super().__init__("explorer")

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.cmd_pub    = self.create_publisher(TwistStamped, "/cmd_vel", qos)
        self.marker_pub = self.create_publisher(MarkerArray, "/exploration_markers", 10)

        self.create_subscription(PointCloud2, "/line_detector/blue", self._line_cb, 10)
        lidar_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                               history=HistoryPolicy.KEEP_LAST, depth=5)
        self.create_subscription(LaserScan, "/scan", self._lidar_cb, lidar_qos)

        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Nav2 action client (no internal spin, no executor conflicts)
        self._nav_action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self._use_nav2 = True
        self._nav_goal_handle = None
        self.get_logger().info("Nav2 action client initialized.")

        # Nav goal state
        self._nav_goal_sent = False
        self._last_goal = None
        self._last_goal_time = 0.0  # time of last goal send
        self._goal_throttle_s = 0.5  # min time between goal sends
        self._turn_nav_sent = False

        # Sensor state
        self.line_pts       = None
        self.last_line_time = None
        self.obstacle_dist  = float("inf")
        self._obs_history   = []
        self._line_cache    = deque()

        # T-junction debounce counter
        self._t_junction_count = 0
        self._branch_target_yaw = None  # angle to best detected branch
        self._last_turn_time = None     # time of last T-junction turn
        self._in_escape_mode = False    # True if backing from obstacle post-turn
        self._recovery_waypoint = None  # (x, y, yaw) for opposite branch fallback
        self._recovery_armed_until = None

        # Control state
        self.state         = "FOLLOWING"
        self.turn_target_rad = np.pi   # radians to turn (pi=180, pi/2=90)
        self.turn_target_yaw = None    # absolute target yaw (if set)
        self.turn_progress = 0.0
        self.turn_last_yaw = None

        self.visited = []
        self.pois    = []

        self.create_timer(0.05, self._loop)
        self.get_logger().info("Explorer ready.")

    # -----------------------------------------------------------------------
    # Callbacks
    # -----------------------------------------------------------------------

    def _line_cb(self, msg: PointCloud2):
        if msg.width == 0:
            return
        buf = np.frombuffer(msg.data, dtype=np.float32).reshape(-1, 4)
        self.line_pts       = buf[:, :3].copy()
        self.last_line_time = self.get_clock().now()

    def _lidar_cb(self, msg: LaserScan):
        ranges = np.array(msg.ranges)
        angles = (msg.angle_min
                  + np.arange(len(ranges)) * msg.angle_increment
                  + LIDAR_ANGLE_OFFSET)
        valid = (ranges > 0.18) & (ranges < msg.range_max)
        if not valid.any():
            self.obstacle_dist = float("inf")
            return
        lx = ranges[valid] * np.cos(angles[valid])
        ly = ranges[valid] * np.sin(angles[valid])
        in_box = (lx > 0.10) & (lx < 0.55) & (np.abs(ly) < OBSTACLE_BOX_W_M)
        hits   = lx[in_box]
        raw = float(np.min(hits)) if len(hits) >= MIN_OBSTACLE_PTS else float("inf")
        self._obs_history.append(raw)
        if len(self._obs_history) > 5:
            self._obs_history.pop(0)
        self.obstacle_dist = min(self._obs_history)

    # -----------------------------------------------------------------------
    # Geometry helpers
    # -----------------------------------------------------------------------

    def _get_pose(self):
        try:
            tf  = self.tf_buffer.lookup_transform(
                      "map", "base_link", Time(),
                      timeout=Duration(seconds=0.05))
            q   = tf.transform.rotation
            yaw = np.arctan2(2*(q.w*q.z + q.x*q.y),
                             1 - 2*(q.y*q.y + q.z*q.z))
            return tf.transform.translation.x, tf.transform.translation.y, yaw
        except Exception:
            return None

    def _to_local(self, global_pts, rx, ry, ryaw):
        dx = global_pts[:, 0] - rx
        dy = global_pts[:, 1] - ry
        c, s = np.cos(ryaw), np.sin(ryaw)
        return np.column_stack((dx*c + dy*s, -dx*s + dy*c))

    def _norm_angle(self, a):
        return np.arctan2(np.sin(a), np.cos(a))

    def _maybe_store_recovery_waypoint(self, local_pts, rx, ry, ryaw, now_s):
        """Store a short-lived waypoint on the right branch before taking left.
        Used if an immediate obstacle blocks the chosen left branch.
        """
        if local_pts is None or len(local_pts) < Y_MIN_PTS:
            self._recovery_waypoint = None
            self._recovery_armed_until = None
            return

        right = local_pts[(local_pts[:, 0] > -0.05) & (local_pts[:, 1] < -0.05)]
        if len(right) < Y_MIN_PTS:
            self._recovery_waypoint = None
            self._recovery_armed_until = None
            return

        # Use cluster center as fallback target and orient along local branch angle.
        cx = float(np.mean(right[:, 0]))
        cy = float(np.mean(right[:, 1]))
        local_yaw = np.arctan2(cy, max(cx, 1e-3))
        c, s = np.cos(ryaw), np.sin(ryaw)
        gx = rx + cx*c - cy*s
        gy = ry + cx*s + cy*c
        gyaw = self._norm_angle(ryaw + local_yaw)

        self._recovery_waypoint = (gx, gy, gyaw)
        self._recovery_armed_until = now_s + 1.5
        self.get_logger().info(
            f"Recovery waypoint armed at ({gx:.2f}, {gy:.2f}) for 1.5s")

    def _clear_recovery_waypoint(self):
        self._recovery_waypoint = None
        self._recovery_armed_until = None

    # -----------------------------------------------------------------------
    # Junction detection
    # -----------------------------------------------------------------------

    def _check_t_junction(self, local_pts):
        """
        Returns True if the point cloud looks like a T-junction crossbar:
          - almost no forward extent  (max_x < T_FORWARD_DEPTH_M)
          - wide lateral spread       (y range > T_BAR_MIN_SPREAD_M)
        Uses a debounce counter so transient frames do not trigger a turn.
        Side-effect: updates self._t_junction_count.
        Also tries to estimate the best branch direction (left/right/forward).
        """
        if len(local_pts) < Y_MIN_PTS:
            self._t_junction_count = 0
            return False

        forward_max = float(np.max(local_pts[:, 0]))
        lat_spread  = float(np.max(local_pts[:, 1]) - np.min(local_pts[:, 1]))

        if forward_max < T_FORWARD_DEPTH_M and lat_spread > T_BAR_MIN_SPREAD_M:
            self._t_junction_count += 1
            # Try to predict best branch direction
            if self._t_junction_count == T_CONFIRM_FRAMES - 1:
                self._predict_branch_direction(local_pts)
        else:
            self._t_junction_count = 0

        return self._t_junction_count >= T_CONFIRM_FRAMES

    def _predict_branch_direction(self, local_pts):
        """
        Analyze cached line data to predict which branch direction has the
        strongest/clearest line. Store prediction in self._branch_target_yaw
        for use in the turn.
        """
        self._branch_target_yaw = None
        
        # Look at recent cache to see if we can infer branch directions
        if not self._line_cache or len(self._line_cache) < 2:
            return
        
        try:
            # Get the most recent cached points
            recent_pts = []
            for ts, pts in list(self._line_cache)[-5:]:
                if len(pts) > 0 and len(recent_pts) < 100:
                    recent_pts.extend(pts)
            
            if len(recent_pts) < 10:
                return
            
            recent_pts = np.array(recent_pts)
            
            # Split into left and right clusters based on y-coordinate
            left_pts  = recent_pts[recent_pts[:, 1] > 0.05]
            right_pts = recent_pts[recent_pts[:, 1] < -0.05]
            
            best_direction = None
            best_score = 0
            
            # Left branch: look for points going left-forward
            if len(left_pts) > Y_MIN_PTS:
                left_forward = np.mean(left_pts[:, 0])
                left_lateral = np.mean(left_pts[:, 1])
                left_dist = np.sqrt(left_forward**2 + left_lateral**2)
                if left_dist > 0.2:
                    left_angle = np.arctan2(left_lateral, left_forward)
                    left_score = len(left_pts)
                    if best_score < left_score:
                        best_direction = left_angle
                        best_score = left_score
            
            # Right branch: look for points going right-forward
            if len(right_pts) > Y_MIN_PTS:
                right_forward = np.mean(right_pts[:, 0])
                right_lateral = np.mean(right_pts[:, 1])
                right_dist = np.sqrt(right_forward**2 + right_lateral**2)
                if right_dist > 0.2:
                    right_angle = np.arctan2(right_lateral, right_forward)
                    right_score = len(right_pts)
                    if best_score < right_score:
                        best_direction = right_angle
                        best_score = right_score
            
            if best_direction is not None:
                self._branch_target_yaw = best_direction
                self.get_logger().info(
                    f"T-junction: predicted branch at {np.degrees(best_direction):.1f}° (score={best_score})")
        except Exception as e:
            self.get_logger().debug(f"branch prediction failed: {e}")

    def _y_junction_left_branch(self, pts):
        """
        Detects a Y-junction in `pts` (already filtered to forward points).
        If found, returns only the points belonging to the leftmost branch.
        Returns the original pts unchanged if no Y-junction is found.
        """
        band  = np.abs(pts[:, 0] - Y_PROBE_X_M) < Y_PROBE_W_M
        probe = pts[band]
        if len(probe) < Y_MIN_PTS * 2:
            return pts

        y_sorted    = np.sort(probe[:, 1])
        gaps        = np.diff(y_sorted)
        max_gap_idx = int(np.argmax(gaps))
        if gaps[max_gap_idx] < Y_GAP_M:
            return pts

        split_y       = y_sorted[max_gap_idx]
        left_cluster  = probe[probe[:, 1] >  split_y]
        right_cluster = probe[probe[:, 1] <= split_y]
        if len(left_cluster) < Y_MIN_PTS or len(right_cluster) < Y_MIN_PTS:
            return pts

        # Classify all forward points by nearest cluster centre
        left_cy  = float(np.mean(left_cluster[:, 1]))
        right_cy = float(np.mean(right_cluster[:, 1]))
        dist_left  = np.abs(pts[:, 1] - left_cy)
        dist_right = np.abs(pts[:, 1] - right_cy)
        left_pts   = pts[dist_left <= dist_right]

        self.get_logger().info(
            f"Y junction -- left branch centre y={left_cy:.2f} m")
        return left_pts if len(left_pts) >= 4 else pts

    # -----------------------------------------------------------------------
    # Steering target
    # -----------------------------------------------------------------------

    def _steering_target(self, local_pts):
        """
        Returns (tx, ty) in local frame, or None.
        Handles Y-junctions by isolating the left branch.
        Does NOT handle T-junctions (those are caught in _loop before this).
        """
        forward = local_pts[:, 0] > 0.05
        pts     = local_pts[forward]
        if len(pts) < 4:
            return None

        pts = self._y_junction_left_branch(pts)

        order  = np.argsort(pts[:, 0])
        pts    = pts[order]
        degree = 2 if len(pts) >= 10 else 1
        try:
            coeffs = np.polyfit(pts[:, 0], pts[:, 1], degree)
        except Exception:
            return None

        tx = min(LOOKAHEAD_M, float(pts[-1, 0]))
        ty = float(np.polyval(coeffs, tx))
        if not np.isfinite(ty) or abs(ty) > 1.5:
            return None
        return tx, ty

    # -----------------------------------------------------------------------
    # Line cache
    # -----------------------------------------------------------------------

    def _add_to_cache(self, global_pts_2d, now_s):
        if len(global_pts_2d) == 0:
            return
        self._line_cache.append((now_s, global_pts_2d.copy()))
        # Extend cache lifetime during escape mode to preserve context
        cache_lifetime = LINE_CACHE_S * 2.0 if self._in_escape_mode else LINE_CACHE_S
        while self._line_cache and (now_s - self._line_cache[0][0]) > cache_lifetime:
            self._line_cache.popleft()
        total = sum(len(f[1]) for f in self._line_cache)
        while total > CACHE_MAX_PTS and len(self._line_cache) > 1:
            total -= len(self._line_cache.popleft()[1])

    def _get_cached_local(self, rx, ry, ryaw, now_s):
        frames = [pts for ts, pts in self._line_cache
                  if (now_s - ts) <= LINE_CACHE_S]
        if not frames:
            return None
        return self._to_local(np.vstack(frames), rx, ry, ryaw)

    # -----------------------------------------------------------------------
    # Map update
    # -----------------------------------------------------------------------

    def _update_map(self, rx, ry, ryaw, local_pts):
        from scipy.spatial import cKDTree
        pos = np.array([rx, ry])
        if (not self.visited
                or np.linalg.norm(np.array(self.visited[-1]) - pos)
                   > BREADCRUMB_SPACING_M):
            self.visited.append((rx, ry))
        if len(local_pts) == 0:
            return
        c, s = np.cos(ryaw), np.sin(ryaw)
        gx   = rx + local_pts[:, 0]*c - local_pts[:, 1]*s
        gy   = ry + local_pts[:, 0]*s + local_pts[:, 1]*c
        gpts = np.column_stack((gx, gy))
        if self.pois:
            self.pois = [p for p in self.pois
                         if np.linalg.norm(np.array(p) - pos) > FRONTIER_RADIUS_M]
        candidates = gpts
        if self.visited:
            dists, _ = cKDTree(self.visited).query(gpts)
            candidates = gpts[dists > FRONTIER_RADIUS_M]
        for cand in candidates:
            if not self.pois:
                self.pois.append(tuple(cand))
            else:
                d, _ = cKDTree(self.pois).query(cand)
                if d > POI_MIN_SPACING_M:
                    self.pois.append(tuple(cand))

    # -----------------------------------------------------------------------
    # Control loop
    # -----------------------------------------------------------------------

    def _loop(self):
        pose = self._get_pose()
        if pose is None:
            return
        rx, ry, ryaw = pose
        now   = self.get_clock().now()
        now_s = now.nanoseconds / 1e9
        line_age = ((now - self.last_line_time).nanoseconds / 1e9
                    if self.last_line_time else float("inf"))

        # Expire fallback if left branch was not immediately blocked.
        if (self._recovery_armed_until is not None
                and now_s > self._recovery_armed_until):
            self.get_logger().info("Recovery waypoint expired (left branch clear).")
            self._clear_recovery_waypoint()

        # 1. Ingest fresh detections and push to cache
        fresh_local = None
        if self.line_pts is not None and line_age < FRESH_WINDOW_S:
            fresh_local = self._to_local(self.line_pts[:, :2], rx, ry, ryaw)
            c, s = np.cos(ryaw), np.sin(ryaw)
            gx = rx + fresh_local[:, 0]*c - fresh_local[:, 1]*s
            gy = ry + fresh_local[:, 0]*s + fresh_local[:, 1]*c
            self._add_to_cache(np.column_stack((gx, gy)), now_s)
            self._update_map(rx, ry, ryaw, fresh_local)

        # 2. Best available points = fresh + cached
        cached_local = self._get_cached_local(rx, ry, ryaw, now_s)
        if fresh_local is not None and cached_local is not None:
            active = np.vstack((fresh_local, cached_local))
        elif fresh_local is not None:
            active = fresh_local
        else:
            active = cached_local

        # 3. Safety: LiDAR obstacle -> 180 turn
        # But skip if we just completed a T-junction turn (give robot time to clear)
        time_since_turn = float('inf')
        if self._last_turn_time is not None:
            time_since_turn = now_s - self._last_turn_time
        
        skip_obstacle_check = (self.state == "TURNING" or time_since_turn < 0.5)
        if self.obstacle_dist <= OBSTACLE_DIST_M and not skip_obstacle_check:
            # If obstacle appears right after taking left, go to stored opposite branch.
            if (self._recovery_waypoint is not None
                    and self._recovery_armed_until is not None
                    and now_s <= self._recovery_armed_until):
                gx, gy, gyaw = self._recovery_waypoint
                self.get_logger().warn(
                    "Immediate post-turn obstacle -- navigating to stored recovery waypoint.")
                self._last_goal_time = 0.0  # allow immediate send
                if self._send_goal(gx, gy, gyaw):
                    self._clear_recovery_waypoint()
                    self._publish_markers()
                    return

            self.get_logger().warn("Obstacle -- turning 180 deg.")
            self._in_escape_mode = True
            self._clear_recovery_waypoint()
            self._start_turn(target_rad=np.pi)

        # 4. State machine
        if self.state == "TURNING":
            self._do_turn((rx, ry, ryaw))
            # When turning completes, exit escape mode if active
            if self._in_escape_mode:
                self._in_escape_mode = False
                self.get_logger().info("Escape turn complete.")
            self._publish_markers()
            return

        # FOLLOWING
        have_pts  = active is not None and len(active) >= 4
        line_lost = (not have_pts
                     and not self._line_cache
                     and line_age > LINE_LOST_TIMEOUT_S)

        if line_lost:
            self.get_logger().warn("Line lost -- turning 180 deg.")
            self._clear_recovery_waypoint()
            self._start_turn(target_rad=np.pi)
            self._publish_markers()
            return

        # 5. T-junction check (before steering -- must be done on fresh local
        #    only, not cache, to avoid acting on stale geometry)
        if fresh_local is not None and len(fresh_local) >= Y_MIN_PTS:
            if self._check_t_junction(fresh_local):
                self.get_logger().info("T junction confirmed -- turning left.")
                self._t_junction_count = 0   # reset so it does not re-trigger
                self._last_turn_time = now_s  # track time of this turn for obstacle debounce

                # Save opposite branch as immediate-fallback waypoint.
                self._maybe_store_recovery_waypoint(fresh_local, rx, ry, ryaw, now_s)
                
                # If we predicted a specific branch direction, turn towards it.
                # Otherwise, default to 90-deg left turn.
                if self._branch_target_yaw is not None:
                    self._start_turn(target_yaw=self._branch_target_yaw)
                else:
                    self._start_turn(target_rad=np.pi / 2)
                self._branch_target_yaw = None
                self._in_escape_mode = False  # not escaping, normal turn
                self._publish_markers()
                return
        else:
            # No fresh points this frame -- reset counter so we need a fresh
            # run of confirmed frames before acting
            self._t_junction_count = 0

        if not have_pts:
            self._publish(LINEAR_SPEED * MIN_LINEAR_FRAC, 0.0)
            self._publish_markers()
            return

        target = self._steering_target(active)
        if target is None:
            self._publish(LINEAR_SPEED * MIN_LINEAR_FRAC, 0.0)
            self._publish_markers()
            return

        tx, ty  = target
        ang_err = np.arctan2(ty, tx)
        ang_cmd = float(np.clip(ANGULAR_GAIN * ang_err, -1.2, 1.2))

        if abs(ang_err) >= SPIN_DEADBAND_RAD:
            lin_cmd = 0.0
        else:
            frac    = abs(ang_err) / SPIN_DEADBAND_RAD
            lin_cmd = LINEAR_SPEED * (1.0 - (1.0 - MIN_LINEAR_FRAC) * frac)

        # If Nav2 is enabled, send lookahead pose goals in map frame; otherwise
        # publish direct velocity commands as before.
        if self._use_nav2:
            c, s = np.cos(ryaw), np.sin(ryaw)
            gx = rx + tx*c - ty*s
            gy = ry + tx*s + ty*c
            goal_yaw = self._norm_angle(ryaw + np.arctan2(ty, tx))
            # Only send if the goal has significantly changed
            if (self._last_goal is None
                    or np.hypot(gx - self._last_goal[0], gy - self._last_goal[1]) > 0.10):
                goal_sent = self._send_goal(gx, gy, goal_yaw)
                # If goal was not sent (throttled or navigator busy), use velocity fallback
                if not goal_sent:
                    self._publish(lin_cmd, ang_cmd)
        else:
            self._publish(lin_cmd, ang_cmd)
        self._publish_markers()

    # -----------------------------------------------------------------------
    # Turn helpers
    # -----------------------------------------------------------------------

    def _start_turn(self, target_rad: float = None, target_yaw: float = None):
        """Start a turn.
        Args:
            target_rad: radians to turn (pi = 180 deg, pi/2 = 90 deg left).
            target_yaw: if provided, turn to face this absolute yaw instead
                        of turning by a relative angle.
        """
        self.state         = "TURNING"
        self.turn_target_rad = target_rad  # relative rotation amount
        self.turn_target_yaw = target_yaw  # absolute yaw goal (if set)
        self.turn_progress = 0.0
        self.turn_last_yaw = None

    def _do_turn(self, ryaw):
        # Send rotation goal via Nav2 action if available,
        # otherwise fall back to velocity-based turning.
        if isinstance(ryaw, (tuple, list)):
            rx, ry, ryaw = ryaw
        if self.turn_last_yaw is None:
            self.turn_last_yaw = ryaw

        # Compute target based on whether we have an absolute yaw or relative rotation
        if self.turn_target_yaw is not None:
            # Absolute yaw goal (from branch prediction)
            goal_yaw = self._norm_angle(self.turn_target_yaw)
        else:
            # Relative rotation (traditional: turn 180 or 90 deg)
            goal_yaw = self._norm_angle(ryaw + self.turn_target_rad)

        # Try to use Nav2 for turning
        if self._use_nav2:
            if not self._turn_nav_sent:
                try:
                    rx_val, ry_val = rx, ry
                    goal_sent = self._send_goal(rx_val, ry_val, goal_yaw)
                    if goal_sent:
                        self._turn_nav_sent = True
                except Exception:
                    pass

            # Check if turn is complete
            try:
                from action_msgs.msg import GoalStatus
                if self._nav_goal_handle is not None:
                    status = self._nav_goal_handle.status
                    if status in [GoalStatus.STATUS_SUCCEEDED, GoalStatus.STATUS_CANCELED]:
                        self.state = "FOLLOWING"
                        self._turn_nav_sent = False
                        self._nav_goal_handle = None
                        target_deg = (np.degrees(self.turn_target_rad) if self.turn_target_yaw is None
                                     else np.degrees(goal_yaw))
                        self.get_logger().info(
                            f"Turn complete ({target_deg:.0f}°) -- resuming.")
                        return
            except Exception:
                pass

        # Fallback: publish angular velocity and integrate progress
        delta = abs(self._norm_angle(ryaw - self.turn_last_yaw))
        self.turn_progress += delta
        self.turn_last_yaw = ryaw
        
        # For absolute yaw goals, check if we're close to the target
        if self.turn_target_yaw is not None:
            angle_error = abs(self._norm_angle(ryaw - goal_yaw))
            if angle_error < TURN_TOL_RAD:
                self._publish(0.0, 0.0)
                self.state = "FOLLOWING"
                self._turn_nav_sent = False
                self._nav_goal_handle = None
                self.get_logger().info(
                    f"Turn complete ({np.degrees(goal_yaw):.0f}°) -- resuming.")
                return
        else:
            # For relative turns, check progress
            if self.turn_progress >= self.turn_target_rad - TURN_TOL_RAD:
                self._publish(0.0, 0.0)
                self.state = "FOLLOWING"
                self._turn_nav_sent = False
                self._nav_goal_handle = None
                self.get_logger().info(
                    f"Turn complete ({np.degrees(self.turn_target_rad):.0f}°) -- resuming.")
                return
        
        self._publish(0.0, TURN_SPEED_RAD_S)

    # -----------------------------------------------------------------------
    # Publishers
    # -----------------------------------------------------------------------

    def _publish(self, linear: float, angular: float):
        msg = TwistStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.twist.linear.x  = float(linear)
        msg.twist.angular.z = float(angular)
        self.cmd_pub.publish(msg)

    def _send_goal(self, gx: float, gy: float, gyaw: float):
        """Send a pose goal in the `map` frame using Nav2 action client.
        Throttled to avoid queue overflow: only sends if enough time has passed
        and there's no goal currently executing.
        Returns True if goal was sent, False if throttled or already executing.
        """
        if not self._use_nav2:
            return False
        
        now_s = self.get_clock().now().nanoseconds / 1e9
        time_since_last = now_s - self._last_goal_time
        
        # Throttle: don't send if too soon after last send
        if time_since_last < self._goal_throttle_s:
            return False
        
        # Don't send if a goal is already executing
        if self._nav_goal_handle is not None:
            try:
                from action_msgs.msg import GoalStatus
                status = self._nav_goal_handle.status
                if status in [GoalStatus.STATUS_EXECUTING, GoalStatus.STATUS_ACCEPTED]:
                    return False  # Still executing
            except Exception:
                pass  # If we can't check, proceed anyway
        
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = float(gx)
        pose.pose.position.y = float(gy)
        # Simple yaw->quat (no roll/pitch)
        qz = np.sin(gyaw / 2.0)
        qw = np.cos(gyaw / 2.0)
        pose.pose.orientation.z = float(qz)
        pose.pose.orientation.w = float(qw)
        
        goal = NavigateToPose.Goal()
        goal.pose = pose
        
        try:
            # Send goal asynchronously (non-blocking)
            self._nav_goal_handle = self._nav_action_client.send_goal_async(goal)
            self._nav_goal_sent = True
            self._last_goal = (gx, gy, gyaw)
            self._last_goal_time = now_s
            self.get_logger().debug(f"Goal sent to Nav2: ({gx:.2f}, {gy:.2f}, {np.degrees(gyaw):.1f}°)")
            return True
        except Exception as e:
            self.get_logger().warn(f"Failed to send goal to Nav2: {e}")
            return False


    def _publish_markers(self):
        ma = MarkerArray()
        t  = self.get_clock().now().to_msg()
        if self.visited:
            m = Marker()
            m.header.frame_id = "map"
            m.header.stamp    = t
            m.ns, m.id        = "visited", 0
            m.type            = Marker.POINTS
            m.action          = Marker.ADD
            m.scale.x = m.scale.y = 0.05
            m.color.g = 1.0; m.color.a = 0.8
            for p in self.visited:
                pt = Point()
                pt.x = float(p[0]); pt.y = float(p[1]); pt.z = 0.0
                m.points.append(pt)
            ma.markers.append(m)
        for i, p in enumerate(self.pois):
            m = Marker()
            m.header.frame_id = "map"
            m.header.stamp    = t
            m.ns, m.id        = "pois", i + 1000
            m.type            = Marker.SPHERE
            m.action          = Marker.ADD
            m.pose.position.x = float(p[0])
            m.pose.position.y = float(p[1])
            m.pose.position.z = 0.05
            m.scale.x = m.scale.y = m.scale.z = 0.15
            m.color.r = 1.0; m.color.g = 0.6; m.color.a = 1.0
            ma.markers.append(m)
        if ma.markers:
            self.marker_pub.publish(ma)


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(Explorer())


if __name__ == "__main__":
    main()