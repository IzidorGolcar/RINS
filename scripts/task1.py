#!/usr/bin/env python3
import math
import sys
import time
from collections import deque
from enum import Enum, auto

import random
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import String
from rclpy.qos import (QoSDurabilityPolicy, QoSHistoryPolicy,
                        QoSProfile, QoSReliabilityPolicy)
from visualization_msgs.msg import Marker, MarkerArray

import os
sys.path.insert(0, os.path.dirname(__file__))
from robot_commander import RobotCommander  # noqa: E402


# Distance between parallel rows (and between points within each row).
# 1.5 m works well for arenas up to ~10 × 10 m with OAK-D's ~3.5 m range.
# Increase for faster (but sparser) coverage; decrease for thoroughness.
COVERAGE_SPACING = 0.38 # meters

# Sweep axis for the boustrophedon:
#   'y' – horizontal rows (robot sweeps left-right, rows advance up/down)
#   'x' – vertical columns (robot sweeps up-down, columns advance left-right)
# For tall arenas use 'x'; for wide arenas use 'y'.
SWEEP_AXIS = 'y'

ROBOT_CLEARANCE  = 0.1   

ARENA_X_MIN: float | None =  None
ARENA_X_MAX: float | None =  None
ARENA_Y_MIN: float | None =  None
ARENA_Y_MAX: float | None =  None

NUM_FACES = 8
NUM_RINGS = 4
APPROACH_DIST = 0.25
NAV_WAIT_TIMEOUT_SEC = 35.0
RECOVERY_BACKUP_DIST = 0.25
RECOVERY_TIMEOUT_SEC = 5.0
GREETING_TEXT = "Hello! I found your face. Pleased to meet you!"
RING_GREETING_TEMPLATE = "Hello! I found a {color} ring!"
ESPEAK_SPEED = 140      # words per minute


def _bgr_to_color_name(bgr: tuple[int, int, int]) -> str | None:
    """Map ring color to allowed set or None for ambiguous colors."""
    b, g, r = int(bgr[0]), int(bgr[1]), int(bgr[2])
    mx = max(r, g, b)
    mn = min(r, g, b)

    if mx < 70:
        return 'black'

    # Reject low-saturation non-black colors (usually gray/white false detections).
    if (mx - mn) < 45:
        return None

    if r >= g and r >= b:
        return 'red'
    if g >= r and g >= b:
        return 'green'
    if b >= r and b >= g:
        return 'blue'
    return None



_MAP_QOS = QoSProfile(
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1)

_MARKER_QOS = QoSProfile(
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=10)

class State(Enum):
    SEARCHING   = auto()
    APPROACHING = auto()
    GREETING    = auto()
    DONE        = auto()

class Task1Node(RobotCommander):
    """Extends RobotCommander with boustrophedon search and face greeting."""
    def __init__(self):
        super().__init__(node_name='task1')

        self.declare_parameters('', [
            ('enable_startup_spin', True),
        ])
        self.enable_startup_spin = self.get_parameter('enable_startup_spin').get_parameter_value().bool_value

        self.coverage_waypoints: list[tuple[float, float, float]] = []
        self._map_info = None
        self._map_data: np.ndarray | None = None

        self.known_faces: dict[int, tuple[float, float, float]] = {}
        self.greeted_ids: set[int]   = set()
        self.to_greet: deque[int] = deque()

        self.known_rings: dict[int, tuple[float, float, float]] = {}
        self.ring_colors: dict[int, tuple[int, int, int]] = {}  
        self.greeted_ring_ids: set[int]   = set()
        self.to_greet_rings: deque[int] = deque()

        self.state            = State.SEARCHING
        self.waypoint_idx     = 0
        self.current_face_id: int | None = None
        self.current_ring_id: int | None = None
        self.approach_fail_count: dict[int, int] = {}  # id -> retry count
        self.MAX_APPROACH_RETRIES = 3
        self.waypoint_fail_count = 0
        self.MAX_WAYPOINT_RETRIES = 2

        self.create_subscription(
            OccupancyGrid, '/map', self._map_cb, _MAP_QOS)

        self.create_subscription(
            MarkerArray, '/people_markers', self._people_marker_cb, _MARKER_QOS)

        self.create_subscription(
            MarkerArray, '/ring_markers', self._ring_marker_cb, _MARKER_QOS)

        self.waypoint_marker_pub = self.create_publisher(
            MarkerArray, '/coverage_waypoints', 10)
        self.speak_pub = self.create_publisher(String, '/speak', 10)
        self.create_timer(1.0, self._publish_waypoint_markers)

        self.info('Task1 node ready – waiting for map and Nav2.')

    def _map_cb(self, msg: OccupancyGrid) -> None:
        self._map_info = msg.info
        self._map_data = np.array(msg.data, dtype=np.int8).reshape(
            msg.info.height, msg.info.width)
        if self.coverage_waypoints:
            return      # already computed
        self.coverage_waypoints = self._boustrophedon(msg)
        self.info(
            f'Coverage path ready: {len(self.coverage_waypoints)} waypoints '
            f'(spacing={COVERAGE_SPACING} m).')

    def sort_by_nearest_neighbor(self, coords: list[tuple[float, float]]) -> list[tuple[float, float]]:
        if not coords:
            return []
        if len(coords) == 1:
            return list(coords)

        unvisited = list(coords)
        current = unvisited.pop(random.randrange(len(unvisited)))
        path = [current]

        while unvisited:
            cx, cy = current
            nearest = min(unvisited, key=lambda p: math.hypot(p[0] - cx, p[1] - cy))
            unvisited.remove(nearest)
            path.append(nearest)
            current = nearest

        return path


    def _boustrophedon(self, grid: OccupancyGrid) -> list[tuple[float, float, float]]:
        """Generate a path over the free cells of *grid*.
        Returns a list of (world_x, world_y, yaw_deg) tuples.
        """
        res   = grid.info.resolution
        gw    = grid.info.width
        gh    = grid.info.height
        ox    = grid.info.origin.position.x
        oy    = grid.info.origin.position.y

        data = np.array(grid.data, dtype=np.int8).reshape(gh, gw)

        step = max(1, int(COVERAGE_SPACING / res))
        clearance = max(1, int(ROBOT_CLEARANCE  / res))

        waypoints: list[tuple[float, float]] = []

        def _cell_ok(iy: int, ix: int) -> bool:
            if data[iy, ix] != 0:
                return False
            r = clearance
            patch = data[max(0, iy - r):iy + r + 1,
                         max(0, ix - r):ix + r + 1]
            return not np.any(patch == 100)

        def _in_bounds(wx: float, wy: float) -> bool:
            if ARENA_X_MIN is not None and wx < ARENA_X_MIN:
                return False
            if ARENA_X_MAX is not None and wx > ARENA_X_MAX:
                return False
            if ARENA_Y_MIN is not None and wy < ARENA_Y_MIN:
                return False
            if ARENA_Y_MAX is not None and wy > ARENA_Y_MAX:
                return False
            return True

        if SWEEP_AXIS == 'x':
            col_idx = 0
            for ix in range(step // 2, gw, step):
                iys = list(range(step // 2, gh, step))
                if col_idx % 2 == 1:
                    iys = iys[::-1]
                for iy in iys:
                    wx = ox + ix * res
                    wy = oy + iy * res
                    if _cell_ok(iy, ix) and _in_bounds(wx, wy):
                        waypoints.append((wx, wy))
                col_idx += 1
        else:
            row_idx = 0
            for iy in range(step // 2, gh, step):
                xs = list(range(step // 2, gw, step))
                if row_idx % 2 == 1:
                    xs = xs[::-1]
                for ix in xs:
                    wx = ox + ix * res
                    wy = oy + iy * res
                    if _cell_ok(iy, ix) and _in_bounds(wx, wy):
                        waypoints.append((wx, wy))
                row_idx += 1

        if not waypoints:
            self.warn('Boustrophedon: no free cells found – check map QoS.')
            return []

        waypoints = self.sort_by_nearest_neighbor(waypoints)

        result: list[tuple[float, float, float]] = []
        for i, (wx, wy) in enumerate(waypoints):
            if i < len(waypoints) - 1:
                nx, ny = waypoints[i + 1]
                yaw_deg = math.degrees(math.atan2(ny - wy, nx - wx))
            else:
                yaw_deg = result[-1][2] if result else 0.0
            result.append((wx, wy, yaw_deg))

        return result


    def _publish_waypoint_markers(self) -> None:
        if not self.coverage_waypoints:
            return
        now = self.get_clock().now().to_msg()
        ma = MarkerArray()
        for i, (wx, wy, yaw_deg) in enumerate(self.coverage_waypoints):
            visited = i < self.waypoint_idx
            current = i == self.waypoint_idx

            sphere = Marker()
            sphere.header.frame_id = 'map'
            sphere.header.stamp = now
            sphere.ns = 'waypoints'
            sphere.id = i
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position.x = float(wx)
            sphere.pose.position.y = float(wy)
            sphere.pose.position.z = 0.05
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.12
            if current:
                sphere.color.r, sphere.color.g, sphere.color.b = 1.0, 1.0, 0.0
            elif visited:
                sphere.color.r, sphere.color.g, sphere.color.b = 0.3, 0.3, 0.3
            else:
                sphere.color.r, sphere.color.g, sphere.color.b = 0.0, 0.6, 1.0
            sphere.color.a = 0.9
            ma.markers.append(sphere)

            arrow = Marker()
            arrow.header.frame_id = 'map'
            arrow.header.stamp = now
            arrow.ns = 'waypoint_yaw'
            arrow.id = i
            arrow.type = Marker.ARROW
            arrow.action = Marker.ADD
            arrow.pose.position.x = float(wx)
            arrow.pose.position.y = float(wy)
            arrow.pose.position.z = 0.05
            arrow.pose.orientation = self.YawToQuaternion(math.radians(yaw_deg))
            arrow.scale.x = 0.25
            arrow.scale.y = 0.04
            arrow.scale.z = 0.04
            arrow.color.r, arrow.color.g, arrow.color.b, arrow.color.a = 0.0, 0.8, 0.2, 0.8
            ma.markers.append(arrow)

            label = Marker()
            label.header.frame_id = 'map'
            label.header.stamp = now
            label.ns = 'waypoint_labels'
            label.id = i
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x = float(wx)
            label.pose.position.y = float(wy)
            label.pose.position.z = 0.25
            label.pose.orientation.w = 1.0
            label.scale.z = 0.15
            label.color.r = label.color.g = label.color.b = 1.0
            label.color.a = 1.0
            label.text = str(i + 1)
            ma.markers.append(label)

        self.waypoint_marker_pub.publish(ma)

    def _people_marker_cb(self, msg: MarkerArray) -> None:
        if NUM_FACES <= 0:
            return
        for m in msg.markers:
            if m.ns == 'faces':
                fid = m.id
                if fid in self.known_faces:
                    continue
                pos = (m.pose.position.x, m.pose.position.y, m.pose.position.z)
                self.known_faces[fid] = pos
                if fid not in self.greeted_ids:
                    self.to_greet.append(fid)
                    self.info(
                        f'New face #{fid} queued at '
                        f'({pos[0]:.2f}, {pos[1]:.2f})')


    def _ring_marker_cb(self, msg: MarkerArray) -> None:
        if NUM_RINGS <= 0:
            return
        for m in msg.markers:
            if m.ns == 'confirmed_rings':
                rid = m.id
                if rid in self.known_rings:
                    continue
                pos = (m.pose.position.x, m.pose.position.y, m.pose.position.z)
                # Recover BGR from the marker's RGBA color (stored as r,g,b in [0,1])
                bgr = (
                    int(m.color.b * 255),
                    int(m.color.g * 255),
                    int(m.color.r * 255),
                )
                self.known_rings[rid] = pos
                self.ring_colors[rid] = bgr
                if rid not in self.greeted_ring_ids:
                    color_name = _bgr_to_color_name(bgr)
                    if color_name is None:
                        self.warn(
                            f'Ignoring ring #{rid}: non-target color {bgr}.')
                        continue
                    self.to_greet_rings.append(rid)
                    self.info(
                        f'New {color_name} ring #{rid} queued at '
                        f'({pos[0]:.2f}, {pos[1]:.2f})')


    def _has_nearby_goal(self) -> bool:
        """Return True if any confirmed goal should interrupt navigation."""
        return bool((NUM_FACES > 0 and self.to_greet)
                    or (NUM_RINGS > 0 and self.to_greet_rings))

    def _pop_nearest_goal(self) -> tuple[str, int]:
        """Pop and return ('face', id) or ('ring', id) for the nearest queued goal."""
        if self.current_face_id is not None:
            return ('face', self.current_face_id)
        if self.current_ring_id is not None:
            return ('ring', self.current_ring_id)
        if not (hasattr(self, 'current_pose') and self.current_pose is not None):
            if NUM_FACES <= 0 and NUM_RINGS <= 0:
                raise RuntimeError('No goals queued and both faces/rings are disabled.')
            if NUM_RINGS > 0 and self.to_greet_rings:
                return ('ring', self.to_greet_rings.popleft())
            if NUM_FACES > 0 and self.to_greet:
                return ('face', self.to_greet.popleft())
            if NUM_RINGS <= 0:
                raise RuntimeError('No face goals queued and rings are disabled.')
            return ('ring', self.to_greet_rings.popleft())
        rx = self.current_pose.pose.position.x
        ry = self.current_pose.pose.position.y
        best_dist = float('inf')
        best_type = 'face'
        best_id = -1

        if NUM_RINGS > 0:
            for rid in self.to_greet_rings:
                rx2, ry2, _ = self.known_rings[rid]
                d = math.hypot(rx2 - rx, ry2 - ry)
                if d < best_dist:
                    best_dist, best_type, best_id = d, 'ring', rid

        if NUM_FACES > 0:
            for fid in self.to_greet:
                fx, fy, _ = self.known_faces[fid]
                d = math.hypot(fx - rx, fy - ry)
                if d < best_dist:
                    best_dist, best_type, best_id = d, 'face', fid
        if best_id < 0:
            raise RuntimeError('No eligible goal found for current task settings.')
        if best_type == 'face':
            self.to_greet.remove(best_id)
        else:
            self.to_greet_rings.remove(best_id)
        return (best_type, best_id)

    def _look_at_face(self, face_id: int) -> None:
        """Rotate in place to face the detected face before greeting."""
        if not (hasattr(self, 'current_pose') and self.current_pose is not None):
            return

        fx, fy, _ = self.known_faces[face_id]
        cx = self.current_pose.pose.position.x
        cy = self.current_pose.pose.position.y
        target_yaw = math.atan2(fy - cy, fx - cx)

        q = self.current_pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        current_yaw = math.atan2(siny_cosp, cosy_cosp)

        delta_yaw = math.atan2(
            math.sin(target_yaw - current_yaw),
            math.cos(target_yaw - current_yaw),
        )

        if abs(delta_yaw) > 0.05:
            spin_time = max(3, int(abs(delta_yaw) / 0.8) + 2)
            self.spin(spin_dist=delta_yaw, time_allowance=spin_time)
            self._wait_nav(allow_interrupt=False)

        end_time = time.time() + 0.7
        while time.time() < end_time:
            self._spin_ros(timeout=0.05)

    def _look_at_ring(self, ring_id: int) -> None:
        """Rotate in place to face the detected ring before greeting."""
        if not (hasattr(self, 'current_pose') and self.current_pose is not None):
            return

        rx, ry, _ = self.known_rings[ring_id]
        cx = self.current_pose.pose.position.x
        cy = self.current_pose.pose.position.y
        target_yaw = math.atan2(ry - cy, rx - cx)

        q = self.current_pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        current_yaw = math.atan2(siny_cosp, cosy_cosp)

        delta_yaw = math.atan2(
            math.sin(target_yaw - current_yaw),
            math.cos(target_yaw - current_yaw),
        )

        if abs(delta_yaw) > 0.05:
            spin_time = max(3, int(abs(delta_yaw) / 0.8) + 2)
            self.spin(spin_dist=delta_yaw, time_allowance=spin_time)
            self._wait_nav(allow_interrupt=False)

        end_time = time.time() + 0.4
        while time.time() < end_time:
            self._spin_ros(timeout=0.05)

    def _spin_ros(self, timeout: float = 0.05) -> None:
        rclpy.spin_once(self, timeout_sec=timeout)

    def _recover_from_obstacle(self) -> bool:
        """Back up a bit and yield control to planner for a new route."""
        if not (hasattr(self, 'current_pose') and self.current_pose is not None):
            return False

        cx = self.current_pose.pose.position.x
        cy = self.current_pose.pose.position.y
        q = self.current_pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        retreat = PoseStamped()
        retreat.header.frame_id = 'map'
        retreat.header.stamp = self.get_clock().now().to_msg()
        retreat.pose.position.x = cx - RECOVERY_BACKUP_DIST * math.cos(yaw)
        retreat.pose.position.y = cy - RECOVERY_BACKUP_DIST * math.sin(yaw)
        retreat.pose.position.z = 0.0
        retreat.pose.orientation = self.YawToQuaternion(yaw)

        self.warn('Navigation blocked: backing up and trying alternate path.')
        if not self.goToPose(retreat):
            return False

        start = time.time()
        while not self.isTaskComplete():
            self._spin_ros(timeout=0.05)
            if (time.time() - start) >= RECOVERY_TIMEOUT_SEC:
                self.cancelTask()
                return False

        return True

    def _wait_nav(self, allow_interrupt: bool = True) -> bool:
        """Spin until current Nav2 goal finishes.
        """
        start_time = time.time()
        while not self.isTaskComplete():
            self._spin_ros()
            if allow_interrupt and self._has_nearby_goal():
                self.cancelTask()
                self.info('Navigation cancelled – new target queued.')
                return False

            if (time.time() - start_time) >= NAV_WAIT_TIMEOUT_SEC:
                self.cancelTask()
                self.warn(
                    f'Navigation timed out after {NAV_WAIT_TIMEOUT_SEC:.1f}s; '
                    'task cancelled.')
                return False

        # Check the actual result status – treat anything other than SUCCEEDED as failure
        result = self.getResult()
        if result and str(result) != 'TaskResult.SUCCEEDED':
            self.warn(f'Navigation finished with non-success result: {result}')
            return False

        return True

    def _go_waypoint(self, x: float, y: float, yaw_deg: float) -> None:
        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.header.stamp    = self.get_clock().now().to_msg()
        goal.pose.position.x = x
        goal.pose.position.y = y
        goal.pose.orientation = self.YawToQuaternion(math.radians(yaw_deg))
        self.goToPose(goal)

    def _approach_pose(self, face_id: int) -> PoseStamped:
        fx, fy, fz = self.known_faces[face_id]
        return self._approach_pose_for(fx, fy, fz)

    def _approach_pose_ring(self, ring_id: int) -> PoseStamped:
        rx2, ry2, rz2 = self.known_rings[ring_id]
        return self._approach_pose_for(rx2, ry2, rz2)

    def _clearance_at(self, wx: float, wy: float) -> float:
        """Return distance (m) to nearest obstacle at world position (wx, wy).
        Returns 0 if outside map or on obstacle, large value if fully clear."""
        if self._map_info is None or self._map_data is None:
            return float('inf')
        res = self._map_info.resolution
        ox  = self._map_info.origin.position.x
        oy  = self._map_info.origin.position.y
        ix  = int((wx - ox) / res)
        iy  = int((wy - oy) / res)
        gh, gw = self._map_data.shape
        if not (0 <= ix < gw and 0 <= iy < gh):
            return 0.0
        if self._map_data[iy, ix] != 0:
            return 0.0
        # Expand a square patch until we hit an obstacle or the edge
        for r in range(1, max(gw, gh)):
            y0, y1 = max(0, iy - r), min(gh, iy + r + 1)
            x0, x1 = max(0, ix - r), min(gw, ix + r + 1)
            patch = self._map_data[y0:y1, x0:x1]
            if np.any(patch == 100):
                return r * res
        return float('inf')

    def _approach_pose_for(self, fx: float, fy: float, fz: float) -> PoseStamped:
        if hasattr(self, 'current_pose') and self.current_pose is not None:
            rx = self.current_pose.pose.position.x
            ry = self.current_pose.pose.position.y
        else:
            rx, ry = 0.0, 0.0

        dx, dy = rx - fx, ry - fy
        length = math.hypot(dx, dy)
        base_angle = math.atan2(dy, dx) if length >= 1e-3 else 0.0

        # Try 12 candidate angles (every 30°), pick the one with most clearance
        best_ax, best_ay, best_clearance = None, None, -1.0
        for i in range(12):
            angle = base_angle + i * (math.pi / 6)
            cdx, cdy = math.cos(angle), math.sin(angle)
            ax = fx + cdx * APPROACH_DIST
            ay = fy + cdy * APPROACH_DIST
            c = self._clearance_at(ax, ay)
            if c > best_clearance:
                best_clearance = c
                best_ax, best_ay = ax, ay

        ax, ay = best_ax, best_ay
        yaw = math.atan2(fy - ay, fx - ax)

        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = ax
        goal.pose.position.y = ay
        goal.pose.position.z = 0.0
        goal.pose.orientation = self.YawToQuaternion(yaw)
        return goal


    def _say(self, text: str) -> None:
        self.info(f'Speaking: "{text}"')
        msg = String()
        msg.data = text
        self.speak_pub.publish(msg)
        words = len(text.split())
        wait_sec = (words / ESPEAK_SPEED) * 60.0 + 1.5
        end_time = time.time() + wait_sec
        while time.time() < end_time:
            self._spin_ros(timeout=0.05)

    def _greet(self) -> None:
        self._say(GREETING_TEXT)


    def run(self) -> None:
        self.waitUntilNav2Active()
        
        # Perform startup spin to improve localization if enabled
        if self.enable_startup_spin:
            self.info('Performing 360° startup spin to improve localization...')
            self.spin(spin_dist=2 * math.pi, time_allowance=20)
            if not self._wait_nav(allow_interrupt=False):
                self.warn('Startup spin was interrupted or failed; continuing anyway.')
            else:
                self.info('Startup spin completed.')
        
        self.info('Waiting for /map to build coverage path...')
        while not self.coverage_waypoints and rclpy.ok():
            self._spin_ros(0.1)

        if hasattr(self, 'current_pose') and self.current_pose is not None:
            rx = self.current_pose.pose.position.x
            ry = self.current_pose.pose.position.y
            dists = [math.hypot(wx - rx, wy - ry)
                     for wx, wy, _ in self.coverage_waypoints]
            start = int(np.argmin(dists))
            self.coverage_waypoints = (self.coverage_waypoints[start:]
                                       + self.coverage_waypoints[:start])
            self.info(f'Path starts at waypoint {start + 1} '
                      f'(nearest to spawn, '
                      f'd={dists[start]:.2f} m).')

        self.info(
            f'Starting search over {len(self.coverage_waypoints)} waypoints.')
        self._say("Let's start")

        while rclpy.ok():

            if self.state == State.DONE:
                self.info(
                    f'All {NUM_FACES} faces and {NUM_RINGS} rings greeted. '
                    f'Task complete!')
                self._say("I'm done, that was easy")
                break

            elif self.state == State.SEARCHING:

                if self._has_nearby_goal():
                    self.state = State.APPROACHING
                    continue

                if self.waypoint_idx >= len(self.coverage_waypoints):
                    # Path done — approach any deferred goals before restarting
                    if ((NUM_FACES > 0 and self.to_greet)
                            or (NUM_RINGS > 0 and self.to_greet_rings)):
                        self.state = State.APPROACHING
                        continue
                    faces_done = len(self.greeted_ids) >= NUM_FACES
                    rings_done = len(self.greeted_ring_ids) >= NUM_RINGS
                    if faces_done and rings_done:
                        self.state = State.DONE
                    else:
                        self.info(
                            'Path exhausted; restarting '
                            f'({len(self.greeted_ids)}/{NUM_FACES} faces, '
                            f'{len(self.greeted_ring_ids)}/{NUM_RINGS} rings greeted).')
                        self.waypoint_idx = 0
                    continue

                wp = self.coverage_waypoints[self.waypoint_idx]
                self.info(
                    f'Waypoint {self.waypoint_idx + 1}/'
                    f'{len(self.coverage_waypoints)}: '
                    f'({wp[0]:.2f}, {wp[1]:.2f}) '
                    f'yaw={wp[2]:.0f}°')
                self._go_waypoint(*wp)

                if not self._wait_nav(allow_interrupt=True):
                    if self._has_nearby_goal():
                        self.waypoint_idx += 1
                        self.waypoint_fail_count = 0
                        self.state = State.APPROACHING
                    else:
                        self._recover_from_obstacle()
                        self.waypoint_fail_count += 1
                        if self.waypoint_fail_count >= self.MAX_WAYPOINT_RETRIES:
                            self.warn(
                                f'Skipping unreachable waypoint {self.waypoint_idx + 1} '
                                f'at ({wp[0]:.2f}, {wp[1]:.2f}).')
                            self.waypoint_idx += 1
                            self.waypoint_fail_count = 0
                        else:
                            # Probe a different route by advancing to next waypoint.
                            self.waypoint_idx = min(
                                self.waypoint_idx + 1,
                                len(self.coverage_waypoints)
                            )
                else:
                    self.waypoint_idx += 1
                    self.waypoint_fail_count = 0
                    if self._has_nearby_goal():
                        self.state = State.APPROACHING

            elif self.state == State.APPROACHING:

                if self.current_face_id is None and self.current_ring_id is None:
                    if not ((NUM_FACES > 0 and self.to_greet)
                            or (NUM_RINGS > 0 and self.to_greet_rings)):
                        self.state = State.SEARCHING
                        continue

                    goal_type, goal_id = self._pop_nearest_goal()

                    if goal_type == 'face':
                        if goal_id in self.greeted_ids:
                            continue
                        self.current_face_id = goal_id
                        self.current_ring_id = None
                        fx, fy, _ = self.known_faces[goal_id]
                        self.info(
                            f'Approaching face #{goal_id} at ({fx:.2f}, {fy:.2f}).')
                        self.goToPose(self._approach_pose(goal_id))
                    else:
                        if goal_id in self.greeted_ring_ids:
                            continue
                        self.current_ring_id = goal_id
                        self.current_face_id = None
                        rx2, ry2, _ = self.known_rings[goal_id]
                        color_name = _bgr_to_color_name(self.ring_colors[goal_id]) or 'black'
                        self.info(
                            f'Approaching {color_name} ring #{goal_id} '
                            f'at ({rx2:.2f}, {ry2:.2f}).')
                        self.goToPose(self._approach_pose_ring(goal_id))

                if self.current_face_id is None and self.current_ring_id is None:
                    self.state = State.SEARCHING
                    continue

                if self.current_face_id is not None:
                    goal_type = 'face'
                    goal_id = self.current_face_id
                else:
                    goal_type = 'ring'
                    goal_id = self.current_ring_id

                completed = self._wait_nav(allow_interrupt=False)
                if completed:
                    self.state = State.GREETING
                else:
                    self._recover_from_obstacle()
                    retries = self.approach_fail_count.get(goal_id, 0) + 1
                    self.approach_fail_count[goal_id] = retries
                    if retries >= self.MAX_APPROACH_RETRIES:
                        self.warn(
                            f'Giving up on {goal_type} #{goal_id} after '
                            f'{retries} failed attempts.')
                        if goal_type == 'face':
                            self.greeted_ids.add(goal_id)
                        else:
                            self.greeted_ring_ids.add(goal_id)
                    else:
                        self.warn(
                            f'Failed to reach {goal_type} #{goal_id} '
                            f'(attempt {retries}/{self.MAX_APPROACH_RETRIES}). '
                            f'Re-queuing.')
                        if goal_type == 'face':
                            self.to_greet.append(goal_id)
                        else:
                            self.to_greet_rings.append(goal_id)
                    self.current_face_id = None
                    self.current_ring_id = None
                    self.state = State.SEARCHING

            elif self.state == State.GREETING:

                if self.current_face_id is not None:
                    face_id = self.current_face_id
                    self._look_at_face(face_id)
                    self._greet()
                    self.greeted_ids.add(face_id)
                    self.current_face_id = None
                    self.info(
                        f'Greeted face #{face_id}.  '
                        f'Total faces: {len(self.greeted_ids)}/{NUM_FACES}.')

                elif self.current_ring_id is not None:
                    ring_id = self.current_ring_id
                    self._look_at_ring(ring_id)
                    color_name = _bgr_to_color_name(self.ring_colors[ring_id]) or 'black'
                    text = RING_GREETING_TEMPLATE.format(color=color_name)
                    self._say(text)
                    self.greeted_ring_ids.add(ring_id)
                    self.current_ring_id = None
                    self.info(
                        f'Greeted {color_name} ring #{ring_id}.  '
                        f'Total rings: {len(self.greeted_ring_ids)}/{NUM_RINGS}.')

                faces_done = len(self.greeted_ids) >= NUM_FACES
                rings_done = len(self.greeted_ring_ids) >= NUM_RINGS
                self.state = (State.DONE
                              if faces_done and rings_done
                              else State.SEARCHING)

            self._spin_ros()


def main() -> None:
    print('Task 1 node starting.')
    rclpy.init(args=None)
    node = Task1Node()
    try:
        node.run()
    finally:
        node.destroyNode()
        rclpy.shutdown()


if __name__ == '__main__':
    main()