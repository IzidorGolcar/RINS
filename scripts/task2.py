#!/usr/bin/env python3
"""Task 2 (Industry 5.0) orchestrator.

RobotCommander subclass that:
  * Boustrophedon-explores the first room (coverage path generated
    from /map) until every named worker has been greeted.
  * On finding a worker, approaches them, spins to face them, then
    runs a dialogue exchange via the dialogue_node (Vosk STT + TTS +
    QR fallback). The dialogue node publishes /dialogue/intent which
    we wait for.
  * Dispatches the chosen task: barrels (visit each horizontal barrel
    for spill confirmation), rings (count + colour-classify from
    /ring_markers), or anomaly_red/_green (drive to the cell pose,
    sweep belt with arm wrist camera).
  * Aggregates results into an InspectionReport that is written as
    PDF (reportlab) or Markdown after meeting the CTO.

Topic contract:
  Sub  /map, /people_markers, /recognized_people,
       /barrel_markers, /barrel_inspections, /ring_markers,
       /tile_markers, /dialogue/intent,
       /lines/{yellow_alert,blue_target,cell_detected,
              red_cell_pose,green_cell_pose}
  Pub  /cmd_vel_nav, /arm_command,
       /dialogue/{prompt,say}, /inspection/path
"""

import json
import math
import os
import re
import sys
import threading
import time
from collections import deque
from enum import Enum, auto

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import OccupancyGrid
from rclpy.qos import (QoSDurabilityPolicy, QoSHistoryPolicy,
                        QoSProfile, QoSReliabilityPolicy,
                        qos_profile_sensor_data)
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, String
from visualization_msgs.msg import MarkerArray

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'report'))
from robot_commander import RobotCommander  # noqa: E402
from task1 import COVERAGE_SPACING, ROBOT_CLEARANCE, SWEEP_AXIS  # noqa: E402
from inspection_report import (  # noqa: E402
    BarrelEntry, InspectionReport, RingsSummary, TileEntry,
)


# ---- Tunables -------------------------------------------------------------

APPROACH_DIST = 0.30        # m — stand-off when greeting a worker. Must be < (face MIN_DIST=0.5 - nav2 xy_goal_tolerance=0.22) so even a face confirmed at min range forces a visible move.
# Positive = counter-clockwise (look further left). The QR card sits to the
# left of the face from the viewer's POV, so aiming a bit left of the face
# brings the QR into the OAK-D's FoV for both dialogue + standalone QR scan.
FACE_YAW_BIAS_DEG = 30.0
SAFETY_BACKUP_DIST = 0.08   # m — reverse this far when /yellow_alert fires
SAFETY_BACKUP_VEL  = -0.10  # m/s
BLUE_FOLLOW_TIMEOUT = 2.0   # s — stop following if line vanishes for longer
BLUE_GOAL_REPLAN_PERIOD = 0.5  # s — how often we re-publish a blue follow goal
EXIT_FIRST_ROOM_GOAL = (3.0, -0.5, -90.0)   # (x, y, yaw_deg) — blue-line entrance
CTO_NAME = 'jeff'  # see personnel/jeff_he_him_cto.png
FACES_TO_VISIT = 3          # hardcoded count of workers in the first room
                            # (we leave once all are greeted; some may
                            # decline a task and that's fine)

DIALOGUE_TIMEOUT_S = 30.0   # how long to wait for /dialogue/intent
BELT_SWEEP_DIST = 1.8       # m — distance to drive along the conveyor belt
BELT_SWEEP_VEL  = 0.08      # m/s — slow so anomaly_detector has frames to lock


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
        self.greeted_names: set[str] = set()       # dedup across face-id flicker
        self.to_greet: deque[int] = deque()
        self._approach_retries: dict[int, int] = {}   # fid -> failed-nav count (for logging)
        self._current_face_id: int | None = None
        self._cto_face_id: int | None = None
        self._chosen_task: str | None = None
        self._chosen_task_requestor: str | None = None
        self._tasks_executed: int = 0  # counts real task runs (not 'nothing')

        # Line state
        self._yellow_alert = False
        self._last_yellow_alert_at = 0.0
        self._blue_target: PoseStamped | None = None
        self._last_blue_target_at = 0.0
        self._cell_seen: str | None = None
        self._red_cell_pose: PoseStamped | None = None
        self._green_cell_pose: PoseStamped | None = None
        # Per-frame line-follow targets used to orient along the cell stripe
        # during anomaly inspection (base_link-frame PoseStamped).
        self._red_target: PoseStamped | None = None
        self._green_target: PoseStamped | None = None
        self._last_red_target_at = 0.0
        self._last_green_target_at = 0.0

        # Blue-search spin state: when blue line is lost while heading to
        # CTO, we spin alternately to re-acquire instead of just pausing.
        self._blue_search_attempts = 0

        # Latest LaserScan, used by _scoot_to_belt during anomaly inspection.
        self._last_scan: LaserScan | None = None

        # Detector beliefs (populated from marker topics).
        self.barrels: dict[int, dict] = {}     # barrel id -> dict from /barrel_inspections
        self.rings:   dict[int, dict] = {}     # ring id   -> {position, color_rgb}
        self.tiles:   dict[int, dict] = {}     # tile id   -> {position, anomalous}

        # Dialogue exchange state.
        self._latest_intent: dict | None = None
        self._intent_event = threading.Event()

        # Inspection report aggregator.
        self.report = InspectionReport(robot_name='R2D2')

        self.declare_parameters('', [
            # False = after coverage + all tasks done, EXIT_FIRST_ROOM →
            # FOLLOW_BLUE → CTO → DONE. True = stop after first room.
            ('declare_first_room_only', False),
            ('exit_x', EXIT_FIRST_ROOM_GOAL[0]),
            ('exit_y', EXIT_FIRST_ROOM_GOAL[1]),
            ('exit_yaw_deg', EXIT_FIRST_ROOM_GOAL[2]),
            # Cell entrance fallbacks (x, y, yaw_deg) used when
            # line_detection hasn't published a live pose yet. Overridable
            # at launch with `-p cell_red_xy:='4,0,90'` etc.
            ('cell_red_xy',   '-4.5,0,-90'),
            ('cell_green_xy', '-4.4,-2,-180'),
            # Hardcoded start/end of each colored stripe in MAP frame.
            # Read with RViz "Publish Point" tool. Format: 'x,y'.
            # Inspection: Nav2 to start (facing end), arm out, drive forward.
            ('cell_red_start_xy',    '0.21,-4.90'),
            ('cell_red_end_xy',     '-1.65,-4.90'),
            ('cell_green_start_xy', '-4.85,-2.37'),
            ('cell_green_end_xy',   '-4.85, 0.12'),
            # Yellow-bounded first room. Coverage waypoints get clipped to
            # this AABB so exploration never wanders past the yellow lines.
            ('first_room_x_min', -4.5),
            ('first_room_x_max',  1.4),
            ('first_room_y_min', -4.5),
            ('first_room_y_max',  0.7),
            # Re-entry pose into the first room after each CTO report.
            # Just inside x_max, same y as the exit, facing back west into
            # the room (exit is at +x, room is at -x from there).
            ('reentry_x', 1.0),
            ('reentry_y', -0.5),
            ('reentry_yaw_deg', 180.0),
        ])

        # Subscriptions
        self.create_subscription(OccupancyGrid, '/map', self._map_cb, _MAP_QOS)
        # /people_markers uses TRANSIENT_LOCAL on the publisher (detect_people)
        # so late-joining task2 still picks up faces confirmed before this
        # subscription was set up. Must match the publisher's QoS.
        self.create_subscription(MarkerArray,   '/people_markers',
                                 self._people_marker_cb, _MAP_QOS)
        self.create_subscription(MarkerArray,   '/barrel_markers', self._barrel_marker_cb, 10)
        self.create_subscription(String,        '/barrel_inspections', self._barrel_json_cb, 10)
        self.create_subscription(MarkerArray,   '/ring_markers',  self._ring_marker_cb, 10)
        self.create_subscription(MarkerArray,   '/tile_markers',  self._tile_marker_cb, 10)
        self.create_subscription(Bool,          '/lines/yellow_alert', self._yellow_cb, 10)
        self.create_subscription(PoseStamped,   '/lines/blue_target',  self._blue_cb,   10)
        self.create_subscription(String,        '/lines/cell_detected', self._cell_cb,  10)
        self.create_subscription(PoseStamped,   '/lines/red_cell_pose',
                                 self._red_cell_pose_cb, 10)
        self.create_subscription(PoseStamped,   '/lines/green_cell_pose',
                                 self._green_cell_pose_cb, 10)
        # Per-frame "follow this line" targets used during anomaly inspection.
        self.create_subscription(PoseStamped,   '/lines/red_target',
                                 self._red_target_cb, 10)
        self.create_subscription(PoseStamped,   '/lines/green_target',
                                 self._green_target_cb, 10)
        self.create_subscription(String,        '/recognized_people',  self._known_people_cb, 10)
        self.create_subscription(String,        '/dialogue/intent',    self._dialogue_intent_cb, 10)
        self.create_subscription(LaserScan,     '/scan',               self._scan_cb,
                                 qos_profile_sensor_data)

        # Publishers
        # Direct cmd_vel for the safety backup + belt sweep (Nav2 owns /cmd_vel_nav).
        self.cmd_vel_pub     = self.create_publisher(TwistStamped, '/cmd_vel_nav',     10)
        self.arm_cmd_pub     = self.create_publisher(String,       '/arm_command',     10)
        self.dlg_prompt_pub  = self.create_publisher(String,       '/dialogue/prompt', 10)
        self.dlg_say_pub     = self.create_publisher(String,       '/dialogue/say',    10)
        self.report_path_pub = self.create_publisher(String,       '/inspection/path', _MAP_QOS)

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
                # Refresh position.
                self.known_faces[fid]['pos'] = pos
                # Race fix: if /recognized_people landed first and added the
                # face to known_faces without queueing, queue it now.
                self._enqueue_if_pending(fid, self.known_faces[fid]['name'])
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

            recognised = bool(name)
            if not recognised:
                # Spec p.15 shortcut: approach unrecognised faces too,
                # using a fid-based placeholder so the dedup logic still
                # works. Dialogue defaults to neutral phrasing.
                name = f'worker_{fid}'

            self.known_faces[fid] = {'pos': pos, 'name': name, 'role': role,
                                      'gender': gender, 'recognised': recognised}
            if recognised and name == CTO_NAME:
                self._cto_face_id = fid
                self.info(f'CTO ({name}) registered as face #{fid}.')
            tag = name if recognised else f'unknown face #{fid}'
            self._enqueue_if_pending(fid, name, log_tag=f'{tag} ({role or "?"})')

    def _enqueue_if_pending(self, fid: int, name: str | None,
                             log_tag: str | None = None) -> None:
        """Queue this face for approach unless already greeted / queued."""
        if not name:
            self.info(f'Skipping face #{fid}: no name yet.')
            return
        if fid in self.greeted_ids:
            return
        # Closes the race between popleft() in APPROACH_PERSON and
        # greeted_ids.add() in DIALOGUE: while the robot is mid-approach
        # the face is in neither container, so without this guard a fresh
        # marker would re-queue it and we'd visit the same face twice.
        if self._current_face_id == fid:
            return
        if name in self.greeted_names:
            self.info(f'Skipping face #{fid} ({name}): name already greeted.')
            return
        if fid in self.to_greet:
            return
        self.to_greet.append(fid)
        self.info(f'Queued for approach: {log_tag or f"face #{fid} ({name})"} '
                  f'(state={self.state.name}, queue={len(self.to_greet)})')

    def _yellow_cb(self, msg: Bool) -> None:
        self._yellow_alert = bool(msg.data)
        if self._yellow_alert:
            self._last_yellow_alert_at = time.monotonic()

    def _blue_cb(self, msg: PoseStamped) -> None:
        self._blue_target = msg
        self._last_blue_target_at = time.monotonic()

    def _scan_cb(self, msg: LaserScan) -> None:
        self._last_scan = msg

    def _cell_cb(self, msg: String) -> None:
        self._cell_seen = msg.data

    def _red_cell_pose_cb(self, msg: PoseStamped) -> None:
        self._red_cell_pose = msg

    def _red_target_cb(self, msg: PoseStamped) -> None:
        self._red_target = msg
        self._last_red_target_at = time.monotonic()

    def _green_target_cb(self, msg: PoseStamped) -> None:
        self._green_target = msg
        self._last_green_target_at = time.monotonic()

    def _green_cell_pose_cb(self, msg: PoseStamped) -> None:
        self._green_cell_pose = msg

    def _barrel_marker_cb(self, msg: MarkerArray) -> None:
        for m in msg.markers:
            if m.ns != 'confirmed_barrels':
                continue
            entry = self.barrels.setdefault(m.id, {})
            entry['position'] = (m.pose.position.x, m.pose.position.y, m.pose.position.z)

    def _barrel_json_cb(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except (ValueError, TypeError):
            return
        if not isinstance(payload, list):
            return
        for entry in payload:
            bid = int(entry.get('id', -1))
            if bid < 0:
                continue
            self.barrels[bid] = entry  # full JSON snapshot replaces local store

    def _ring_marker_cb(self, msg: MarkerArray) -> None:
        for m in msg.markers:
            if m.ns != 'confirmed_rings':
                continue
            self.rings[m.id] = {
                'position': (m.pose.position.x, m.pose.position.y, m.pose.position.z),
                'color_rgb': (m.color.r, m.color.g, m.color.b),
            }

    def _tile_marker_cb(self, msg: MarkerArray) -> None:
        for m in msg.markers:
            if m.ns != 'tiles':
                continue
            # anomaly_detector encodes status via marker colour (red == anomalous).
            anomalous = m.color.r > 0.5 and m.color.g < 0.5
            self.tiles[m.id] = {
                'position': (m.pose.position.x, m.pose.position.y, m.pose.position.z),
                'anomalous': bool(anomalous),
            }

    def _dialogue_intent_cb(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except (ValueError, TypeError):
            return
        self._latest_intent = payload
        self._intent_event.set()

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
            # Always try to queue — if /recognized_people arrives before
            # /people_markers, this is what triggers the approach.
            self._enqueue_if_pending(fid, name, log_tag=f'{name} (via /recognized_people)')

    # ---- Coverage path generation (lifted from task1.py) -------------------

    def _boustrophedon(self, grid: OccupancyGrid) -> list[tuple[float, float, float]]:
        res = grid.info.resolution
        gw, gh = grid.info.width, grid.info.height
        ox = grid.info.origin.position.x
        oy = grid.info.origin.position.y
        data = np.array(grid.data, dtype=np.int8).reshape(gh, gw)

        step = max(1, int(COVERAGE_SPACING / res))
        clearance = max(1, int(ROBOT_CLEARANCE / res))

        # Yellow-bounded first room AABB. Cells outside this box are skipped
        # so the autonomous coverage path stays inside the yellow lines.
        rx_min = float(self.get_parameter('first_room_x_min').get_parameter_value().double_value)
        rx_max = float(self.get_parameter('first_room_x_max').get_parameter_value().double_value)
        ry_min = float(self.get_parameter('first_room_y_min').get_parameter_value().double_value)
        ry_max = float(self.get_parameter('first_room_y_max').get_parameter_value().double_value)

        waypoints: list[tuple[float, float]] = []

        def cell_ok(iy: int, ix: int) -> bool:
            if data[iy, ix] != 0:
                return False
            wx = ox + ix * res
            wy = oy + iy * res
            if not (rx_min <= wx <= rx_max and ry_min <= wy <= ry_max):
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

    def _approach_pose(self, fx: float, fy: float,
                        distance: float = APPROACH_DIST) -> PoseStamped:
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
            ax = fx + math.cos(angle) * distance
            ay = fy + math.sin(angle) * distance
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
        """Speak via the dialogue node (fire-and-forget; node serialises TTS)."""
        if not text:
            return
        self.info(f'Speaking: "{text}"')
        self.dlg_say_pub.publish(String(data=text))

    # ---- Dialogue exchange -------------------------------------------------

    def _do_dialogue(self, face: dict, face_id: int | None) -> str | None:
        """Run one prompt → /dialogue/intent round-trip. Returns the intent or None."""
        gender_word = 'man' if face.get('gender') == 'male' else 'woman'
        text = f'Hi {gender_word}, which task should I perform?'
        payload = {
            'text': text,
            'gender': face.get('gender'),
            'face_id': face_id,
            'expects_intent': True,
        }
        self._latest_intent = None
        self._intent_event.clear()
        self.dlg_prompt_pub.publish(String(data=json.dumps(payload)))
        if not self._wait_for_intent(timeout=DIALOGUE_TIMEOUT_S):
            self.warn('Dialogue timed out with no intent.')
            return None
        return (self._latest_intent or {}).get('intent')

    def _wait_for_intent(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and rclpy.ok():
            self._spin_ros(0.1)
            if self._intent_event.is_set():
                return True
        return False

    @staticmethod
    def _intent_to_words(intent: str) -> str:
        return {
            'barrels':       'inspect the barrels',
            'rings':         'count the rings',
            'anomaly_red':   'inspect anomalies in the red cell',
            'anomaly_green': 'inspect anomalies in the green cell',
        }.get(intent, intent.replace('_', ' '))

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

        # Park the arm folded back ('garage') so it isn't sticking out
        # during navigation — only the anomaly inspection extends it.
        self._arm('garage')

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
                # Primary exit condition: all workers greeted. Task count
                # doesn't matter — some workers may decline ('nothing').
                if len(self.greeted_ids) >= FACES_TO_VISIT:
                    self.info(f'All {FACES_TO_VISIT} workers visited '
                              f'({self._tasks_executed} tasks ran) — '
                              'leaving first room for CTO report.')
                    if first_room_only:
                        self.state = State.DONE
                    else:
                        self.state = State.EXIT_FIRST_ROOM
                    continue
                if self.waypoint_idx >= len(self.coverage_waypoints):
                    if first_room_only:
                        # Path exhausted with fewer than N faces — give up.
                        self.warn(f'Coverage exhausted but only '
                                  f'{len(self.greeted_ids)}/{FACES_TO_VISIT} '
                                  f'faces visited; finishing anyway.')
                        self.state = State.DONE
                        continue
                    self.warn(f'Coverage exhausted but only '
                              f'{len(self.greeted_ids)}/{FACES_TO_VISIT} '
                              f'faces visited; heading to CTO anyway.')
                    self.state = State.EXIT_FIRST_ROOM
                    continue

                wp = self.coverage_waypoints[self.waypoint_idx]
                self.info(f'Coverage waypoint {self.waypoint_idx + 1}/'
                          f'{len(self.coverage_waypoints)}: ({wp[0]:.2f}, {wp[1]:.2f})')
                self._go_waypoint(*wp)

                # Interrupt the coverage leg the moment a known face is queued
                # — _wait_nav_with_safety only watches yellow_alert.
                interrupted_by_face = False
                while not self.isTaskComplete():
                    self._spin_ros()
                    if self._check_yellow_safety():
                        break
                    if self.to_greet:
                        self.info('Face spotted mid-leg — cancelling coverage to greet.')
                        self.cancelTask()
                        interrupted_by_face = True
                        break

                if not interrupted_by_face and not self._yellow_alert:
                    self.waypoint_idx += 1

            elif self.state == State.APPROACH_PERSON:
                if not self.to_greet:
                    self.state = State.EXPLORE_FIRST_ROOM
                    continue
                fid = self.to_greet.popleft()
                # Defensive: if a stale entry somehow lingers in to_greet
                # (e.g. older code revision had a race), don't re-greet.
                if fid in self.greeted_ids:
                    self.info(f'Face #{fid} already greeted; skipping.')
                    self.state = State.EXPLORE_FIRST_ROOM
                    continue
                self._current_face_id = fid
                face = self.known_faces[fid]
                fx, fy, _ = face['pos']
                attempt = self._approach_retries.get(fid, 0) + 1
                # Back off 0.2 m per failed attempt so a face in a tight
                # corner eventually gets a reachable approach pose.
                dist = APPROACH_DIST + 0.2 * (attempt - 1)
                # Diagnostic so we can tell when the approach pose was
                # already inside Nav2's xy_goal_tolerance (= robot skips
                # the visible drive and goes straight to dialogue).
                rx_dbg = ry_dbg = 0.0
                if self.current_pose is not None:
                    rx_dbg = self.current_pose.pose.position.x
                    ry_dbg = self.current_pose.pose.position.y
                face_dist = math.hypot(fx - rx_dbg, fy - ry_dbg)
                approach_pose = self._approach_pose(fx, fy, distance=dist)
                ap_x = approach_pose.pose.position.x
                ap_y = approach_pose.pose.position.y
                gap = math.hypot(ap_x - rx_dbg, ap_y - ry_dbg)
                self.info(
                    f'Approaching {face["name"]} ({face["role"]}) — '
                    f'face=({fx:.2f},{fy:.2f}) robot=({rx_dbg:.2f},{ry_dbg:.2f}) '
                    f'face_dist={face_dist:.2f}m '
                    f'approach=({ap_x:.2f},{ap_y:.2f}) '
                    f'gap_to_approach={gap:.2f}m (nav2 xy tol=0.22m) '
                    f'attempt={attempt} stand-off={dist:.2f}m')
                self.goToPose(approach_pose)
                if self._wait_nav_with_safety():
                    self._approach_retries.pop(fid, None)
                    self._face_person(fx, fy)
                    self.state = State.DIALOGUE
                else:
                    # Re-queue at the END so other faces get tried first.
                    self._approach_retries[fid] = attempt
                    self.warn(
                        f'Approach to face #{fid} failed (attempt {attempt}); '
                        f're-queued — will retry after other work.')
                    self.to_greet.append(fid)
                    self.state = State.EXPLORE_FIRST_ROOM

            elif self.state == State.DIALOGUE:
                fid = self._current_face_id
                face = self.known_faces.get(fid, {}) if fid is not None else {}
                name = face.get('name', 'there')

                intent = self._do_dialogue(face, fid)
                self._chosen_task = intent
                self._chosen_task_requestor = name.capitalize() if name and name != 'there' else 'unknown'
                if intent and intent != 'nothing':
                    self._say(f'OK {name}, I will {self._intent_to_words(intent)}.')
                    self.state = State.EXECUTE_TASK
                else:
                    self._say('OK, never mind then.')
                    self.state = State.EXPLORE_FIRST_ROOM

                if fid is not None:
                    self.greeted_ids.add(fid)
                if name and name != 'there':
                    self.greeted_names.add(name)
                self._current_face_id = None

            elif self.state == State.EXECUTE_TASK:
                intent = self._chosen_task
                requestor = self._chosen_task_requestor or 'unknown'
                task_was_run = False
                if intent == 'barrels':
                    self._run_barrel_inspection(requestor)
                    task_was_run = True
                elif intent == 'rings':
                    self._run_ring_counting(requestor)
                    task_was_run = True
                elif intent == 'anomaly_red':
                    self._run_anomaly_inspection('red', requestor)
                    task_was_run = True
                elif intent == 'anomaly_green':
                    self._run_anomaly_inspection('green', requestor)
                    task_was_run = True
                else:
                    self.info(f'EXECUTE_TASK: unrecognised intent {intent!r}; skipping.')
                if task_was_run:
                    self._tasks_executed += 1
                    self.info(f'Tasks executed: {self._tasks_executed}; '
                              f'faces greeted: {len(self.greeted_ids)}/{FACES_TO_VISIT}.')
                self._chosen_task = None
                self._chosen_task_requestor = None
                # Batched flow: finish all worker tasks first, then ONE CTO
                # trip at the end. Always go back to exploration after a task.
                self.state = State.EXPLORE_FIRST_ROOM

            elif self.state == State.EXIT_FIRST_ROOM:
                # Catch late-discovered faces before committing to CTO trip.
                if self.to_greet:
                    self.info(f'Late face queued ({len(self.to_greet)}); '
                              f'handling before exit.')
                    self.state = State.APPROACH_PERSON
                    continue
                ex = float(self.get_parameter('exit_x').get_parameter_value().double_value)
                ey = float(self.get_parameter('exit_y').get_parameter_value().double_value)
                eyaw = float(self.get_parameter('exit_yaw_deg').get_parameter_value().double_value)
                self.info(f'All worker tasks done — heading to exit ({ex:.2f}, {ey:.2f}) for CTO.')
                self._go_waypoint(ex, ey, eyaw)
                interrupted_by_face = False
                while not self.isTaskComplete():
                    self._spin_ros()
                    if self._check_yellow_safety():
                        break
                    if self.to_greet:
                        self.info('Face spotted on the way out — handling first.')
                        self.cancelTask()
                        interrupted_by_face = True
                        break
                if interrupted_by_face:
                    self.state = State.APPROACH_PERSON
                elif self._yellow_alert:
                    self.state = State.EXIT_FIRST_ROOM
                else:
                    self.state = State.FOLLOW_BLUE_LINE

            elif self.state == State.FOLLOW_BLUE_LINE:
                # A face confirmed after leaving the first room still needs a
                # greeting + dialogue. "If not doing task" → preempt FOLLOW.
                if self.to_greet:
                    self.info(f'Face queued during FOLLOW_BLUE_LINE '
                              f'({len(self.to_greet)}); handling first.')
                    self.cancelTask()
                    self.state = State.APPROACH_PERSON
                    continue
                if self._cto_face_id is not None and self._cto_face_id in self.known_faces:
                    self.info('CTO already in sight – approaching.')
                    self.state = State.REPORT_TO_CTO
                    continue

                if (self._blue_target is None or
                        time.monotonic() - self._last_blue_target_at > BLUE_FOLLOW_TIMEOUT):
                    self._search_for_blue_line()
                    continue
                # Got the line back — reset the search counter.
                self._blue_search_attempts = 0

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
                    if self.to_greet:
                        self.info('Face spotted while following blue line — '
                                  'cancelling to greet.')
                        self.cancelTask()
                        self.state = State.APPROACH_PERSON
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
                self._face_person(fx, fy)

                # End-of-run: deliver the aggregated inspection report.
                self._finalize_report()
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

    # ---- Sub-task execution (called from EXECUTE_TASK) ---------------------

    def _current_yaw(self) -> float | None:
        if not (hasattr(self, 'current_pose') and self.current_pose is not None):
            return None
        q = self.current_pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def _face_person(self, fx: float, fy: float) -> None:
        """Final small spin so the front camera + mic point at the person."""
        if not (hasattr(self, 'current_pose') and self.current_pose is not None):
            return
        rx = self.current_pose.pose.position.x
        ry = self.current_pose.pose.position.y
        # Aim slightly left of the face so the QR card (to the left of the
        # worker from the camera's POV) enters frame too.
        target_yaw = math.atan2(fy - ry, fx - rx) + math.radians(FACE_YAW_BIAS_DEG)
        cur_yaw = self._current_yaw()
        if cur_yaw is None:
            return
        delta = math.atan2(math.sin(target_yaw - cur_yaw),
                           math.cos(target_yaw - cur_yaw))
        if abs(delta) < math.radians(5):
            return
        self.spin(spin_dist=float(delta), time_allowance=5)
        # Wait for spin to finish (or yellow safety to fire).
        while not self.isTaskComplete():
            self._spin_ros()
            if self._check_yellow_safety():
                return

    def _arm(self, pose: str) -> None:
        self.arm_cmd_pub.publish(String(data=pose))

    def _run_barrel_inspection(self, requestor: str) -> None:
        if not self.barrels:
            self._say('I have not detected any barrels.')
            self.report.add_execution(requestor, 'barrels', [])
            return

        # Park the arm down so detect_barrels.py's top_camera pipeline can
        # confirm spills under horizontal barrels.
        self._arm('look_for_qr')

        # Iterate barrels in id order for deterministic reporting.
        for bid in sorted(self.barrels.keys()):
            entry = self.barrels[bid]
            pos = entry.get('position')
            orientation = entry.get('orientation', 'ambiguous')
            if not pos or orientation == 'vertical':
                # Vertical barrels can't leak; no need to approach.
                continue
            bx, by, _ = pos
            self.info(f'Approaching barrel #{bid} at ({bx:.2f}, {by:.2f}) for spill check.')
            self.goToPose(self._approach_pose(bx, by))
            if not self._wait_nav_with_safety():
                continue
            # Linger so detect_barrels accumulates confirmation frames.
            t_end = time.monotonic() + 2.0
            while time.monotonic() < t_end and rclpy.ok():
                self._spin_ros(0.1)
            updated = self.barrels.get(bid, entry)
            if updated.get('leaking'):
                self._say('Alert! Alert! This barrel is leaking!')

        # Park the arm back up so it doesn't drag.
        self._arm('garage')

        # Snapshot the final state into the report.
        results = []
        for bid in sorted(self.barrels.keys()):
            e = self.barrels[bid]
            results.append(BarrelEntry(
                id=bid,
                colour=str(e.get('colour', 'unknown')),
                orientation=str(e.get('orientation', 'unknown')),
                leaking=bool(e.get('leaking', False)),
            ))
        self.report.add_execution(requestor, 'barrels', results)
        self._say(f'I inspected {len(results)} barrels.')

    def _run_ring_counting(self, requestor: str) -> None:
        per_colour: dict[str, int] = {}
        for r in self.rings.values():
            cr, cg, cb = r.get('color_rgb', (0.5, 0.5, 0.5))
            colour = self._classify_ring_colour(cr, cg, cb)
            per_colour[colour] = per_colour.get(colour, 0) + 1
        total = sum(per_colour.values())
        summary = RingsSummary(total=total, per_colour=per_colour)
        self.report.add_execution(requestor, 'rings', summary)

        if total == 0:
            self._say('I have not seen any rings.')
            return
        parts = ', '.join(f'{n} {c}' for c, n in sorted(per_colour.items()))
        self._say(f'I counted {total} rings: {parts}.')

    @staticmethod
    def _classify_ring_colour(r: float, g: float, b: float) -> str:
        # Compact discriminator over the spec palette.
        if max(r, g, b) < 0.25:
            return 'black'
        if r > 0.6 and g < 0.4 and b < 0.4:
            return 'red'
        if g > 0.5 and r < 0.5 and b < 0.5:
            return 'green'
        if b > 0.5 and r < 0.5 and g < 0.7:
            return 'blue'
        if r > 0.7 and g > 0.6 and b < 0.4:
            return 'yellow'
        if r > 0.7 and g > 0.3 and g < 0.6 and b < 0.3:
            return 'orange'
        if r > 0.4 and b > 0.4 and g < 0.4:
            return 'purple'
        if r > 0.3 and g > 0.2 and b < 0.2:
            return 'brown'
        return 'unknown'

    def _resolve_cell_pose(self, colour: str) -> PoseStamped | None:
        # 1. Live topic from line_detection (friend's work).
        topic_pose = self._red_cell_pose if colour == 'red' else self._green_cell_pose
        if topic_pose is not None:
            return topic_pose
        # 2. Handcoded parameter fallback (spec p.15 permits this).
        param = self.get_parameter(f'cell_{colour}_xy').get_parameter_value().string_value
        if param.strip():
            try:
                parts = [float(x) for x in param.split(',')]
                if len(parts) >= 2:
                    yaw_deg = parts[2] if len(parts) > 2 else 0.0
                    ps = PoseStamped()
                    ps.header.frame_id = 'map'
                    ps.header.stamp = self.get_clock().now().to_msg()
                    ps.pose.position.x = parts[0]
                    ps.pose.position.y = parts[1]
                    ps.pose.orientation = self.YawToQuaternion(math.radians(yaw_deg))
                    return ps
            except ValueError:
                self.warn(f'Bad cell_{colour}_xy parameter: {param!r}')
        return None

    def _run_anomaly_inspection(self, cell_colour: str, requestor: str) -> None:
        # Read hardcoded start/end points (map frame).
        start_xy = self._parse_xy_param(f'cell_{cell_colour}_start_xy')
        end_xy   = self._parse_xy_param(f'cell_{cell_colour}_end_xy')
        if start_xy is None or end_xy is None:
            self._say(f'I do not know where the {cell_colour} stripe is.')
            self.report.add_execution(requestor, f'anomaly_{cell_colour}', [])
            return

        sx, sy = start_xy
        ex, ey = end_xy
        drive_yaw = math.atan2(ey - sy, ex - sx)
        stripe_len = math.hypot(ex - sx, ey - sy)
        self.info(f'{cell_colour} stripe: start=({sx:.2f},{sy:.2f}) '
                  f'end=({ex:.2f},{ey:.2f}) yaw={math.degrees(drive_yaw):.1f}° '
                  f'length={stripe_len:.2f} m')

        # 1) STOP AT START POINT. Nav2 navigates to the start; goal yaw is
        #    set to drive_yaw so Nav2 will try to leave us already facing
        #    the end, but we do an explicit spin in step 2 to guarantee
        #    alignment regardless of how Nav2 finishes.
        start_pose = PoseStamped()
        start_pose.header.frame_id = 'map'
        start_pose.header.stamp = self.get_clock().now().to_msg()
        start_pose.pose.position.x = sx
        start_pose.pose.position.y = sy
        start_pose.pose.orientation = self.YawToQuaternion(drive_yaw)
        self.info(f'Driving to {cell_colour} stripe start.')
        self.goToPose(start_pose)
        if not self._wait_nav_with_safety():
            self.report.add_execution(requestor, f'anomaly_{cell_colour}', [])
            return

        # 2) YAW TOWARD END POINT. Explicit spin so the robot is guaranteed
        #    to face along the stripe before the arm comes out.
        cur_yaw = self._current_yaw() or 0.0
        delta = math.atan2(math.sin(drive_yaw - cur_yaw),
                           math.cos(drive_yaw - cur_yaw))
        self.info(f'Yawing toward end of {cell_colour} stripe '
                  f'(delta={math.degrees(delta):.1f}°).')
        if abs(delta) > math.radians(2):
            self.spin(spin_dist=float(delta), time_allowance=8)
            while not self.isTaskComplete() and rclpy.ok():
                self._spin_ros(0.05)
        # Brief settle.
        settle_deadline = time.monotonic() + 0.4
        while time.monotonic() < settle_deadline and rclpy.ok():
            self._spin_ros(0.05)

        # Clear tiles from any previous run so we report only this cell.
        self.tiles.clear()

        # 3) EXTEND ARM toward the tiles. arm_mover_actions has a 1 Hz
        #    timer + 3 s trajectory `time_from_start`, so each pose
        #    change can take ~4 s end-to-end. Wait 4 s after `up` and
        #    5 s after `look_at_belt_left` so the arm is FULLY in the
        #    scanning position before the robot starts moving.
        self._arm('up')
        time.sleep(4.0)
        self._arm('look_at_belt_left')
        time.sleep(9.0)

        # 4) DRIVE TO END POINT via cmd_vel — straight line guaranteed.
        #    Nav2 would curve to dodge nearby obstacles even when the
        #    direct path is clear. The tight goal tolerance in step 1
        #    plus the explicit spin in step 2 ensure we start exactly
        #    on the stripe facing the right way, so open-loop forward
        #    drive stays on the line. anomaly_detector keeps populating
        #    self.tiles in the background while we travel.
        self._drive_distance(stripe_len, velocity=BELT_SWEEP_VEL,
                              cell_colour=cell_colour)

        # Hold the arm in place for 3 s so anomaly_detector gets a few
        # extra frames of the final tile (otherwise the last tile can be
        # missed because the robot moved past it just as it locked).
        self.info('Stripe end reached; holding 3 s for last-tile lock.')
        end_settle = time.monotonic() + 3.0
        while time.monotonic() < end_settle and rclpy.ok():
            self._spin_ros(0.1)

        self._arm('garage')

        # Snapshot into the report.
        results = [
            TileEntry(id=tid, anomalous=bool(t.get('anomalous', False)))
            for tid, t in sorted(self.tiles.items())
        ]
        self.report.add_execution(requestor, f'anomaly_{cell_colour}', results)
        anomalous = sum(1 for t in results if t.anomalous)
        self._say(f'Inspected {len(results)} tiles in the {cell_colour} cell. '
                  f'{anomalous} appear damaged.')

    def _parse_xy_param(self, name: str) -> tuple[float, float] | None:
        raw = self.get_parameter(name).get_parameter_value().string_value
        try:
            parts = [float(p.strip()) for p in raw.split(',')]
            if len(parts) >= 2:
                return parts[0], parts[1]
        except ValueError:
            pass
        self.warn(f'Bad parameter {name}={raw!r}; expected "x,y".')
        return None

    def _drive_distance(self, target_distance: float, velocity: float,
                         cell_colour: str | None = None,
                         max_time: float | None = None) -> None:
        """Open-loop forward drive of `target_distance` metres in map frame.

        Sole stop condition is the measured map-frame distance travelled
        (per the user's spec: don't rely on lidar / yellow alert here).
        `max_time` defaults to `target_distance / velocity * 3 + 10` so
        the watchdog scales with stripe length and we never time out on
        a long stripe.
        """
        if self.current_pose is None:
            self.warn('No current_pose; cannot drive distance.')
            return
        if velocity <= 0:
            return
        if max_time is None:
            max_time = target_distance / velocity * 3.0 + 10.0

        start_x = self.current_pose.pose.position.x
        start_y = self.current_pose.pose.position.y

        twist = TwistStamped()
        twist.header.frame_id = 'base_link'
        twist.twist.linear.x = velocity
        deadline = time.monotonic() + max_time
        while time.monotonic() < deadline and rclpy.ok():
            twist.header.stamp = self.get_clock().now().to_msg()
            self.cmd_vel_pub.publish(twist)
            self._spin_ros(0.05)
            travelled = math.hypot(
                self.current_pose.pose.position.x - start_x,
                self.current_pose.pose.position.y - start_y)
            if travelled >= target_distance:
                self.info(f'Reached stripe end '
                          f'({travelled:.2f} m / {target_distance:.2f} m).')
                break
        else:
            travelled = math.hypot(
                self.current_pose.pose.position.x - start_x,
                self.current_pose.pose.position.y - start_y)
            self.warn(f'Drive timed out after {max_time:.1f} s '
                      f'({travelled:.2f} m / {target_distance:.2f} m).')
        twist.twist.linear.x = 0.0
        twist.header.stamp = self.get_clock().now().to_msg()
        self.cmd_vel_pub.publish(twist)

    def _search_for_blue_line(self) -> None:
        """Spin to re-acquire the blue line.

        Alternates direction and grows the arc each attempt so a full
        circle is swept in ~3 tries. Stops as soon as a fresh
        /lines/blue_target arrives (the callback runs during _spin_ros).
        """
        self._blue_search_attempts += 1
        n = self._blue_search_attempts
        # 1st: 90° left, 2nd: 180° right, 3rd: 270° left, then loop.
        sign = 1.0 if n % 2 == 1 else -1.0
        magnitude = math.radians(min(90 * n, 270))
        spin_amt = sign * magnitude
        self.warn(f'Lost blue line — search spin {n} '
                  f'({math.degrees(spin_amt):+.0f}°).')
        self.spin(spin_dist=float(spin_amt), time_allowance=8)
        deadline = time.monotonic() + 8.0
        while not self.isTaskComplete() and time.monotonic() < deadline and rclpy.ok():
            self._spin_ros(0.1)
            if self._check_yellow_safety():
                self.cancelTask()
                return
            # Early exit the moment line_detection publishes a fresh target.
            if (self._last_blue_target_at > 0 and
                    time.monotonic() - self._last_blue_target_at < 0.5):
                self.info('Blue line re-acquired during search.')
                self.cancelTask()
                return

    def _front_lidar_distance(self, half_arc_deg: float = 12.0) -> float | None:
        """Closest lidar return in a small cone straight ahead, or None."""
        scan = self._last_scan
        if scan is None or not scan.ranges:
            return None
        half_arc = math.radians(half_arc_deg)
        n = len(scan.ranges)
        a0 = scan.angle_min
        da = scan.angle_increment
        min_d = float('inf')
        for i in range(n):
            angle = a0 + i * da
            # Normalise to [-pi, pi] just in case.
            angle = math.atan2(math.sin(angle), math.cos(angle))
            if abs(angle) > half_arc:
                continue
            r = scan.ranges[i]
            if scan.range_min <= r <= scan.range_max and math.isfinite(r):
                if r < min_d:
                    min_d = r
        return min_d if min_d < float('inf') else None

    def _scoot_to_belt(self, target_distance: float = 0.25,
                        max_advance_s: float = 20.0) -> None:
        """Open-loop creep forward until lidar reads `target_distance` ahead.

        Run after `goToPose(cell_pose)` puts the robot facing the belt.
        Bounded by max_advance_s so we never wander far if lidar misses.
        Default tuned for the case where the hardcoded approach pose
        leaves the robot up to ~2 m from the belt edge.
        """
        # Wait briefly for a fresh scan.
        wait_deadline = time.monotonic() + 1.5
        while self._last_scan is None and time.monotonic() < wait_deadline and rclpy.ok():
            self._spin_ros(0.1)
        if self._last_scan is None:
            self.warn('Lidar silent — skipping belt scoot.')
            return

        twist = TwistStamped()
        twist.header.frame_id = 'base_link'
        twist.twist.linear.x = 0.27  # creep, but fast enough to close 2 m in time
        deadline = time.monotonic() + max_advance_s
        publish_rate = 0.05
        while time.monotonic() < deadline and rclpy.ok():
            self._spin_ros(publish_rate)
            if self._check_yellow_safety():
                break
            d = self._front_lidar_distance()
            if d is not None and d <= target_distance:
                self.info(f'Belt close enough (lidar={d:.2f} m).')
                break
            twist.header.stamp = self.get_clock().now().to_msg()
            self.cmd_vel_pub.publish(twist)
        # Hard stop.
        twist.twist.linear.x = 0.0
        twist.header.stamp = self.get_clock().now().to_msg()
        self.cmd_vel_pub.publish(twist)

    def _approach_cell_line(self, cell_colour: str,
                             stop_ahead: float = 0.05,
                             max_advance_s: float = 12.0) -> None:
        """Creep forward until the red/green stripe is `stop_ahead` m in front.

        Uses /lines/{red,green}_target (base_link frame). The published pose
        is foot + 0.5 m along the fitted-line direction, so we recover the
        foot (closest point on the line to the robot) and watch its forward
        component. Stops on yellow safety or when lidar reports an obstacle
        closer than `stop_ahead`, whichever fires first.
        """
        target_attr = '_red_target' if cell_colour == 'red' else '_green_target'
        seen_attr   = '_last_red_target_at' if cell_colour == 'red' else '_last_green_target_at'
        LOOKAHEAD = 0.5  # must match line_detection._publish_line_target

        twist = TwistStamped()
        twist.header.frame_id = 'base_link'
        twist.twist.linear.x = 0.27  # 1.5x of original creep
        deadline = time.monotonic() + max_advance_s
        publish_rate = 0.05
        line_ever_seen = False

        while time.monotonic() < deadline and rclpy.ok():
            self._spin_ros(publish_rate)
            if self._check_yellow_safety():
                break

            target: PoseStamped | None = getattr(self, target_attr)
            last_at = getattr(self, seen_attr)
            line_fresh = (target is not None
                          and last_at > 0
                          and time.monotonic() - last_at < 0.5)

            if line_fresh:
                line_ever_seen = True
                q = target.pose.orientation
                yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                                 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
                foot_x = target.pose.position.x - LOOKAHEAD * math.cos(yaw)
                # foot_x is "metres ahead of robot to the line".
                if foot_x <= stop_ahead:
                    self.info(f'At {cell_colour} stripe (foot_x={foot_x:.2f} m).')
                    break

            # Lidar safety / fallback for when the line is briefly lost or
            # not yet acquired. Barrel-radius bound = stop_ahead + clearance.
            d = self._front_lidar_distance()
            if d is not None and d <= max(stop_ahead, 0.30):
                if line_ever_seen:
                    self.info(f'Lidar stop at {d:.2f} m (line momentarily lost).')
                else:
                    self.warn(f'No {cell_colour} stripe seen yet; lidar stop at {d:.2f} m.')
                break

            twist.header.stamp = self.get_clock().now().to_msg()
            self.cmd_vel_pub.publish(twist)

        # Hard stop.
        twist.twist.linear.x = 0.0
        twist.header.stamp = self.get_clock().now().to_msg()
        self.cmd_vel_pub.publish(twist)

    def _align_to_cell_line(self, cell_colour: str, timeout_s: float = 4.0) -> None:
        """Spin in place so the robot's forward axis matches the cell line.

        Waits up to `timeout_s` for line_detection to publish a fresh
        red/green target, then spins by the target's yaw (target is in
        base_link frame, so its yaw == the delta to apply).
        """
        target_attr = '_red_target' if cell_colour == 'red' else '_green_target'
        seen_attr   = '_last_red_target_at' if cell_colour == 'red' else '_last_green_target_at'
        # Force a fresh target — anything older than 1 s is stale.
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline and rclpy.ok():
            last_at = getattr(self, seen_attr)
            if last_at > 0 and time.monotonic() - last_at < 1.0:
                break
            self._spin_ros(0.1)
        target: PoseStamped | None = getattr(self, target_attr)
        if target is None:
            self.warn(f'No {cell_colour} target seen; sweeping with hardcoded yaw.')
            return
        q = target.pose.orientation
        # base_link-frame yaw = delta to spin (forward axis is already X).
        delta = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                            1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        if abs(delta) < math.radians(3):
            return
        self.info(f'Aligning to {cell_colour} line by {math.degrees(delta):.1f}°.')
        self.spin(spin_dist=float(delta), time_allowance=5)
        while not self.isTaskComplete():
            self._spin_ros()
            if self._check_yellow_safety():
                return

    def _belt_sweep(self, distance: float, velocity: float) -> None:
        if velocity == 0:
            return
        duration = abs(distance / velocity)
        deadline = time.monotonic() + duration
        twist = TwistStamped()
        twist.header.frame_id = 'base_link'
        twist.twist.linear.x = velocity
        while time.monotonic() < deadline and rclpy.ok():
            twist.header.stamp = self.get_clock().now().to_msg()
            self.cmd_vel_pub.publish(twist)
            self._spin_ros(0.05)
            if self._check_yellow_safety():
                break
        twist.twist.linear.x = 0.0
        twist.header.stamp = self.get_clock().now().to_msg()
        self.cmd_vel_pub.publish(twist)

    # ---- Final report -----------------------------------------------------

    def _finalize_report(self) -> None:
        out_dir = os.path.expanduser('~/.ros')
        try:
            path = self.report.finalize(out_dir=out_dir)
        except Exception as e:
            self.error(f'Failed to write inspection report: {e}')
            return
        self.info(f'Inspection report saved to {path}')
        self.report_path_pub.publish(String(data=str(path)))


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
