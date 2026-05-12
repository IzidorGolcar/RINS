#!/usr/bin/env python3
"""Task 2 (Industry 5.0) orchestrator.

Builds on task1.py's pattern:
  * RobotCommander subclass drives Nav2 via goToPose
  * Detection nodes publish MarkerArray on /people_markers (and friends);
    we subscribe and react.
Adds:
  * Yellow-line safety stop driven by /lines/yellow_alert.
  * Worker recognition: only approach faces that have been recognised by
    detect_people (i.e. carry a name+role label, not 'Face N').
  * Blue-line following in the second room, driven by /lines/blue_target.
  * Stub task dispatch / dialogue (real ASR + barrels + anomalies are
    separate sub-tasks that get merged later).

Out of scope here: ASR/dialogue, barrel inspection, anomaly model, PDF
report generation, new SLAM map.
"""

import json
import math
import os
import re
import subprocess
import sys
import time
from collections import deque
from enum import Enum, auto

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import OccupancyGrid
from rclpy.qos import (QoSDurabilityPolicy, QoSHistoryPolicy,
                        QoSProfile, QoSReliabilityPolicy)
from std_msgs.msg import Bool, String
from visualization_msgs.msg import MarkerArray

sys.path.insert(0, os.path.dirname(__file__))
from robot_commander import RobotCommander  # noqa: E402
from task1 import COVERAGE_SPACING, ROBOT_CLEARANCE, SWEEP_AXIS  # noqa: E402


# ---- Tunables -------------------------------------------------------------

APPROACH_DIST = 0.7         # m — stand-off when greeting a worker
ESPEAK_SPEED  = 140         # words per minute
SAFETY_BACKUP_DIST = 0.15   # m — reverse this far when /yellow_alert fires
SAFETY_BACKUP_VEL  = -0.10  # m/s
BLUE_FOLLOW_TIMEOUT = 2.0   # s — stop following if line vanishes for longer
BLUE_GOAL_REPLAN_PERIOD = 0.5  # s — how often we re-publish a blue follow goal
EXIT_FIRST_ROOM_GOAL = (4.5, 0.0, 0.0)   # (x, y, yaw_deg) — placeholder; tune
CTO_NAME = 'jeff'  # see personnel/jeff_he_him_cto.png


_MAP_QOS = QoSProfile(
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)


# Marker text format from detect_people.py: "name (role)" or "Face N".
_NAMED_LABEL_RE = re.compile(r'^(?P<name>[a-z]+)\s*\((?P<role>[a-z_]+)\)\s*$', re.IGNORECASE)


class State(Enum):
    EXPLORE_FIRST_ROOM = auto()
    APPROACH_PERSON    = auto()
    DIALOGUE           = auto()
    EXIT_FIRST_ROOM    = auto()
    FOLLOW_BLUE_LINE   = auto()
    EXECUTE_TASK       = auto()
    REPORT_TO_CTO      = auto()
    DONE               = auto()


class Task2Node(RobotCommander):
    def __init__(self):
        super().__init__(node_name='task2')

        # Coverage path & map
        self.coverage_waypoints: list[tuple[float, float, float]] = []
        self._map_info = None
        self._map_data: np.ndarray | None = None
        self.waypoint_idx = 0

        # Faces (named only)
        self.known_faces: dict[int, dict] = {}     # id -> {pos, name, role, gender}
        self.greeted_ids: set[int]   = set()
        self.to_greet: deque[int] = deque()
        self._current_face_id: int | None = None
        self._cto_face_id: int | None = None

        # Line state
        self._yellow_alert = False
        self._last_yellow_alert_at = 0.0
        self._blue_target: PoseStamped | None = None
        self._last_blue_target_at = 0.0
        self._cell_seen: str | None = None

        # Dialogue stub: parameter the user (or another node) can set with the
        # task name to perform when in DIALOGUE state.
        self.declare_parameters('', [
            ('dialogue_response', ''),  # 'barrels' | 'rings' | 'anomaly_red' | 'anomaly_green' | ''
            ('declare_first_room_only', True),
            ('exit_x', EXIT_FIRST_ROOM_GOAL[0]),
            ('exit_y', EXIT_FIRST_ROOM_GOAL[1]),
            ('exit_yaw_deg', EXIT_FIRST_ROOM_GOAL[2]),
        ])

        # Subscriptions
        self.create_subscription(OccupancyGrid, '/map', self._map_cb, _MAP_QOS)
        self.create_subscription(MarkerArray,   '/people_markers', self._people_marker_cb, 10)
        self.create_subscription(Bool,          '/lines/yellow_alert', self._yellow_cb, 10)
        self.create_subscription(PoseStamped,   '/lines/blue_target',  self._blue_cb,   10)
        self.create_subscription(String,        '/lines/cell_detected', self._cell_cb,  10)
        self.create_subscription(String,        '/recognized_people',  self._known_people_cb, 10)

        # Direct cmd_vel for the safety backup (Nav2 owns /cmd_vel_nav).
        self.cmd_vel_pub = self.create_publisher(TwistStamped, '/cmd_vel_nav', 10)

        self.state = State.EXPLORE_FIRST_ROOM

        self.info('Task2 node ready – waiting for map and Nav2.')

    # ----------------------------------------------------------- subscriptions

    def _map_cb(self, msg: OccupancyGrid) -> None:
        self._map_info = msg.info
        self._map_data = np.array(msg.data, dtype=np.int8).reshape(
            msg.info.height, msg.info.width)
        if self.coverage_waypoints:
            return
        self.coverage_waypoints = self._boustrophedon(msg)
        self.info(
            f'Coverage path ready: {len(self.coverage_waypoints)} waypoints '
            f'(spacing={COVERAGE_SPACING} m).')

    def _people_marker_cb(self, msg: MarkerArray) -> None:
        # Build a lookup of label text per face id from this batch.
        labels: dict[int, str] = {}
        identities: dict[int, dict] = {}
        positions: dict[int, tuple[float, float, float]] = {}
        for m in msg.markers:
            if m.ns == 'faces':
                positions[m.id] = (m.pose.position.x, m.pose.position.y, m.pose.position.z)
            elif m.ns == 'face_labels':
                labels[m.id] = m.text or ''
            elif m.ns == 'face_identities':
                try:
                    identities[m.id] = json.loads(m.text)
                except (ValueError, TypeError):
                    pass

        for fid, pos in positions.items():
            if fid in self.known_faces:
                # Update the position in case the detector refined it.
                self.known_faces[fid]['pos'] = pos
                continue

            ident = identities.get(fid) or {}
            name = ident.get('name')
            role = ident.get('role')
            gender = ident.get('gender')

            if not name:
                # Fall back to parsing the human-readable label.
                m = _NAMED_LABEL_RE.match(labels.get(fid, '').strip())
                if m:
                    name = m.group('name').lower()
                    role = m.group('role').lower()

            if not name:
                # Skip unrecognised faces — Task 2 only acts on named workers.
                continue

            self.known_faces[fid] = {'pos': pos, 'name': name, 'role': role, 'gender': gender}
            if name == CTO_NAME:
                self._cto_face_id = fid
                self.info(f'CTO ({name}) registered as face #{fid}.')
            if fid not in self.greeted_ids:
                self.to_greet.append(fid)
                self.info(f'New worker {name} ({role}) queued at face #{fid}.')

    def _yellow_cb(self, msg: Bool) -> None:
        self._yellow_alert = bool(msg.data)
        if self._yellow_alert:
            self._last_yellow_alert_at = time.monotonic()

    def _blue_cb(self, msg: PoseStamped) -> None:
        self._blue_target = msg
        self._last_blue_target_at = time.monotonic()

    def _cell_cb(self, msg: String) -> None:
        self._cell_seen = msg.data

    def _known_people_cb(self, msg: String) -> None:
        # Re-hydrate from detect_people's persistence so we recover after restarts.
        try:
            data = json.loads(msg.data)
        except (ValueError, TypeError):
            return
        for entry in data.get('faces', []):
            fid = int(entry['id'])
            name = entry.get('name')
            if not name:
                continue
            existing = self.known_faces.get(fid)
            if existing is None:
                self.known_faces[fid] = {
                    'pos': (entry.get('x', 0.0), entry.get('y', 0.0), entry.get('z', 0.0)),
                    'name': name,
                    'role': entry.get('role'),
                    'gender': entry.get('gender'),
                }
                if name == CTO_NAME:
                    self._cto_face_id = fid

    # ---- Coverage path generation (lifted from task1.py) -------------------

    def _boustrophedon(self, grid: OccupancyGrid) -> list[tuple[float, float, float]]:
        res = grid.info.resolution
        gw, gh = grid.info.width, grid.info.height
        ox = grid.info.origin.position.x
        oy = grid.info.origin.position.y
        data = np.array(grid.data, dtype=np.int8).reshape(gh, gw)

        step = max(1, int(COVERAGE_SPACING / res))
        clearance = max(1, int(ROBOT_CLEARANCE / res))

        waypoints: list[tuple[float, float]] = []

        def cell_ok(iy: int, ix: int) -> bool:
            if data[iy, ix] != 0:
                return False
            r = clearance
            patch = data[max(0, iy - r):iy + r + 1,
                         max(0, ix - r):ix + r + 1]
            return not np.any(patch == 100)

        if SWEEP_AXIS == 'x':
            col = 0
            for ix in range(step // 2, gw, step):
                iys = list(range(step // 2, gh, step))
                if col % 2 == 1:
                    iys = iys[::-1]
                for iy in iys:
                    if cell_ok(iy, ix):
                        waypoints.append((ox + ix * res, oy + iy * res))
                col += 1
        else:
            row = 0
            for iy in range(step // 2, gh, step):
                ixs = list(range(step // 2, gw, step))
                if row % 2 == 1:
                    ixs = ixs[::-1]
                for ix in ixs:
                    if cell_ok(iy, ix):
                        waypoints.append((ox + ix * res, oy + iy * res))
                row += 1

        if not waypoints:
            return []

        result: list[tuple[float, float, float]] = []
        for i, (wx, wy) in enumerate(waypoints):
            if i + 1 < len(waypoints):
                nx, ny = waypoints[i + 1]
                yaw_deg = math.degrees(math.atan2(ny - wy, nx - wx))
            else:
                yaw_deg = result[-1][2] if result else 0.0
            result.append((wx, wy, yaw_deg))
        return result

    # ---- Helpers shared with task1 -----------------------------------------

    def _spin_ros(self, timeout: float = 0.05) -> None:
        rclpy.spin_once(self, timeout_sec=timeout)

    def _go_waypoint(self, x: float, y: float, yaw_deg: float) -> None:
        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.header.stamp    = self.get_clock().now().to_msg()
        goal.pose.position.x = x
        goal.pose.position.y = y
        goal.pose.orientation = self.YawToQuaternion(math.radians(yaw_deg))
        self.goToPose(goal)

    def _clearance_at(self, wx: float, wy: float) -> float:
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
        for r in range(1, max(gw, gh)):
            y0, y1 = max(0, iy - r), min(gh, iy + r + 1)
            x0, x1 = max(0, ix - r), min(gw, ix + r + 1)
            if np.any(self._map_data[y0:y1, x0:x1] == 100):
                return r * res
        return float('inf')

    def _approach_pose(self, fx: float, fy: float) -> PoseStamped:
        if hasattr(self, 'current_pose') and self.current_pose is not None:
            rx = self.current_pose.pose.position.x
            ry = self.current_pose.pose.position.y
        else:
            rx, ry = 0.0, 0.0

        dx, dy = rx - fx, ry - fy
        base_angle = math.atan2(dy, dx) if math.hypot(dx, dy) >= 1e-3 else 0.0

        best_ax, best_ay, best_clear = None, None, -1.0
        for i in range(12):
            angle = base_angle + i * (math.pi / 6)
            ax = fx + math.cos(angle) * APPROACH_DIST
            ay = fy + math.sin(angle) * APPROACH_DIST
            c = self._clearance_at(ax, ay)
            if c > best_clear:
                best_clear = c
                best_ax, best_ay = ax, ay

        ax, ay = best_ax, best_ay
        yaw = math.atan2(fy - ay, fx - ax)
        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = ax
        goal.pose.position.y = ay
        goal.pose.orientation = self.YawToQuaternion(yaw)
        return goal

    def _say(self, text: str) -> None:
        self.info(f'Speaking: "{text}"')
        try:
            proc = subprocess.Popen(
                ['espeak-ng', '-s', str(ESPEAK_SPEED), text],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            self.warn('espeak-ng not installed; skipping speech.')
            return
        while proc.poll() is None:
            self._spin_ros(0.05)

    # ---- Safety stop -------------------------------------------------------

    def _check_yellow_safety(self) -> bool:
        """Returns True if a safety stop fired (caller should re-plan)."""
        if not self._yellow_alert:
            return False
        self.warn('Yellow line ahead – cancelling goal and backing up.')
        self.cancelTask()
        # Brief reverse for SAFETY_BACKUP_DIST at SAFETY_BACKUP_VEL.
        duration = abs(SAFETY_BACKUP_DIST / SAFETY_BACKUP_VEL)
        deadline = time.monotonic() + duration
        twist = TwistStamped()
        twist.header.frame_id = 'base_link'
        twist.twist.linear.x = SAFETY_BACKUP_VEL
        while time.monotonic() < deadline and rclpy.ok():
            twist.header.stamp = self.get_clock().now().to_msg()
            self.cmd_vel_pub.publish(twist)
            self._spin_ros(0.05)
        twist.twist.linear.x = 0.0
        twist.header.stamp = self.get_clock().now().to_msg()
        self.cmd_vel_pub.publish(twist)
        # Wait until the alert clears so we don't immediately re-trigger.
        for _ in range(40):
            self._spin_ros(0.05)
            if not self._yellow_alert:
                break
        return True

    def _wait_nav_with_safety(self) -> bool:
        """Spin while the current Nav2 goal runs; return True on success."""
        while not self.isTaskComplete():
            self._spin_ros()
            if self._check_yellow_safety():
                return False
        result = self.getResult()
        if result and str(result) != 'TaskResult.SUCCEEDED':
            self.warn(f'Navigation finished non-success: {result}')
            return False
        return True

    # ---- Main loop ---------------------------------------------------------

    def run(self) -> None:
        self.waitUntilNav2Active()

        self.info('Waiting for /map to build coverage path...')
        while not self.coverage_waypoints and rclpy.ok():
            self._spin_ros(0.1)

        # Start the boustrophedon at the waypoint nearest to the spawn pose.
        if hasattr(self, 'current_pose') and self.current_pose is not None:
            rx = self.current_pose.pose.position.x
            ry = self.current_pose.pose.position.y
            dists = [math.hypot(wx - rx, wy - ry) for wx, wy, _ in self.coverage_waypoints]
            start = int(np.argmin(dists))
            self.coverage_waypoints = (self.coverage_waypoints[start:] +
                                       self.coverage_waypoints[:start])

        self.info(f'Starting Task 2 with {len(self.coverage_waypoints)} coverage waypoints.')

        first_room_only = bool(
            self.get_parameter('declare_first_room_only').get_parameter_value().bool_value)

        while rclpy.ok():
            if self.state == State.DONE:
                self._say("All tasks complete. Goodbye.")
                break

            elif self.state == State.EXPLORE_FIRST_ROOM:
                if self.to_greet:
                    self.state = State.APPROACH_PERSON
                    continue
                if self.waypoint_idx >= len(self.coverage_waypoints):
                    if first_room_only:
                        # Path exhausted, no more workers — wrap up.
                        self.state = State.DONE
                        continue
                    self.state = State.EXIT_FIRST_ROOM
                    continue

                wp = self.coverage_waypoints[self.waypoint_idx]
                self.info(f'Coverage waypoint {self.waypoint_idx + 1}/'
                          f'{len(self.coverage_waypoints)}: ({wp[0]:.2f}, {wp[1]:.2f})')
                self._go_waypoint(*wp)
                ok = self._wait_nav_with_safety()
                if ok or not self._yellow_alert:
                    self.waypoint_idx += 1

            elif self.state == State.APPROACH_PERSON:
                if not self.to_greet:
                    self.state = State.EXPLORE_FIRST_ROOM
                    continue
                fid = self.to_greet.popleft()
                self._current_face_id = fid
                face = self.known_faces[fid]
                fx, fy, _ = face['pos']
                self.info(f'Approaching {face["name"]} ({face["role"]}) at ({fx:.2f}, {fy:.2f})')
                self.goToPose(self._approach_pose(fx, fy))
                if self._wait_nav_with_safety():
                    self.state = State.DIALOGUE
                else:
                    # Re-queue and try again later.
                    self.to_greet.append(fid)
                    self.state = State.EXPLORE_FIRST_ROOM

            elif self.state == State.DIALOGUE:
                fid = self._current_face_id
                face = self.known_faces.get(fid, {}) if fid is not None else {}
                name = face.get('name', 'there')
                gender_word = 'man' if face.get('gender') == 'male' else 'woman'
                self._say(f'Hi {gender_word}, which task should I perform?')

                # Stub: read the task from the dialogue_response parameter.
                response = self.get_parameter('dialogue_response').get_parameter_value().string_value
                if response:
                    self._say(f'OK {name}, I will {response.replace("_", " ")}.')
                    self.state = State.EXECUTE_TASK
                else:
                    self.info(f'No dialogue response set for {name}; marking greeted.')
                    self.state = State.EXPLORE_FIRST_ROOM
                if fid is not None:
                    self.greeted_ids.add(fid)
                self._current_face_id = None

            elif self.state == State.EXECUTE_TASK:
                self.info('EXECUTE_TASK is a stub — handed off to a separate sub-task.')
                self.state = State.EXPLORE_FIRST_ROOM

            elif self.state == State.EXIT_FIRST_ROOM:
                ex = float(self.get_parameter('exit_x').get_parameter_value().double_value)
                ey = float(self.get_parameter('exit_y').get_parameter_value().double_value)
                eyaw = float(self.get_parameter('exit_yaw_deg').get_parameter_value().double_value)
                self.info(f'Heading to exit ({ex:.2f}, {ey:.2f})')
                self._go_waypoint(ex, ey, eyaw)
                if self._wait_nav_with_safety():
                    self.state = State.FOLLOW_BLUE_LINE
                else:
                    # Try once more on safety stop.
                    self.state = State.EXIT_FIRST_ROOM

            elif self.state == State.FOLLOW_BLUE_LINE:
                if self._cto_face_id is not None and self._cto_face_id in self.known_faces:
                    self.info('CTO already in sight – approaching.')
                    self.state = State.REPORT_TO_CTO
                    continue

                if (self._blue_target is None or
                        time.monotonic() - self._last_blue_target_at > BLUE_FOLLOW_TIMEOUT):
                    self.warn('Lost the blue line — pausing.')
                    self._spin_ros(0.5)
                    continue

                # Convert the latest base_link target into a map-frame goal so Nav2 can plan to it.
                target = self._latest_blue_goal_in_map()
                if target is None:
                    self._spin_ros(0.2)
                    continue
                self.goToPose(target)
                t0 = time.monotonic()
                while not self.isTaskComplete() and rclpy.ok():
                    self._spin_ros(0.1)
                    if self._check_yellow_safety():
                        break
                    if (self._cto_face_id is not None and
                            self._cto_face_id in self.known_faces):
                        self.cancelTask()
                        self.state = State.REPORT_TO_CTO
                        break
                    if time.monotonic() - t0 > BLUE_GOAL_REPLAN_PERIOD:
                        self.cancelTask()
                        break

            elif self.state == State.REPORT_TO_CTO:
                fid = self._cto_face_id
                if fid is None or fid not in self.known_faces:
                    self.warn('Lost track of CTO; backing to FOLLOW_BLUE_LINE.')
                    self.state = State.FOLLOW_BLUE_LINE
                    continue
                fx, fy, _ = self.known_faces[fid]['pos']
                self.goToPose(self._approach_pose(fx, fy))
                self._wait_nav_with_safety()
                self._say('Inspection report ready, sir.')
                self.state = State.DONE

            self._spin_ros()

    def _latest_blue_goal_in_map(self) -> PoseStamped | None:
        """Re-stamp the base_link blue target as a fresh map-frame goal."""
        if self._blue_target is None:
            return None
        if not (hasattr(self, 'current_pose') and self.current_pose is not None):
            return None

        rx = self.current_pose.pose.position.x
        ry = self.current_pose.pose.position.y
        # current_pose is a PoseWithCovariance.pose
        q = self.current_pose.pose.orientation
        # yaw from quaternion
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        ryaw = math.atan2(siny_cosp, cosy_cosp)

        bx = self._blue_target.pose.position.x
        by = self._blue_target.pose.position.y

        # base_link → map: rotate by ryaw then translate by (rx, ry).
        gx = rx + bx * math.cos(ryaw) - by * math.sin(ryaw)
        gy = ry + bx * math.sin(ryaw) + by * math.cos(ryaw)

        target = PoseStamped()
        target.header.frame_id = 'map'
        target.header.stamp = self.get_clock().now().to_msg()
        target.pose.position.x = gx
        target.pose.position.y = gy
        # Face the goal direction.
        target.pose.orientation = self.YawToQuaternion(math.atan2(gy - ry, gx - rx))
        return target


def main():
    print('Task 2 node starting.')
    rclpy.init(args=None)
    node = Task2Node()
    try:
        node.run()
    finally:
        node.destroyNode()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
