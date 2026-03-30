#!/usr/bin/env python3
"""
Task 1 – autonomous face search and greeting.

Coverage strategy
-----------------
On startup the node reads the live /map occupancy grid and generates a
boustrophedon (lawnmower) coverage path automatically.  The only parameter
you need to tune is COVERAGE_SPACING (row/column pitch in metres).  Smaller
→ denser coverage, more waypoints; larger → faster but risks gaps.

No 360° spin is performed at each waypoint.  The robot faces forward as it
drives, and detect_people.py runs continuously in parallel – faces are
discovered in transit, not just at stop points.

State machine
-------------
  SEARCHING  → drive to the next boustrophedon waypoint
     ↓  new face appears in /people_markers while navigating → cancel leg
  APPROACHING → navigate to APPROACH_DIST metres in front of the face
     ↓
  GREETING   → espeak-ng greeting; mark face done
     ↓
  SEARCHING  → continue from next waypoint
     ↓  all NUM_FACES faces greeted (or path exhausted and re-looped)
  DONE
"""

import math
import os
import subprocess
import sys
from collections import deque
from enum import Enum, auto

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
from rclpy.qos import (QoSDurabilityPolicy, QoSHistoryPolicy,
                        QoSProfile, QoSReliabilityPolicy)
from visualization_msgs.msg import MarkerArray

sys.path.insert(0, os.path.dirname(__file__))
from robot_commander import RobotCommander  # noqa: E402


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Distance between parallel rows (and between points within each row).
# 1.5 m works well for arenas up to ~10 × 10 m with OAK-D's ~3.5 m range.
# Increase for faster (but sparser) coverage; decrease for thoroughness.
COVERAGE_SPACING = 0.9      # metres

# Sweep axis for the boustrophedon:
#   'y' – horizontal rows (robot sweeps left-right, rows advance up/down)
#   'x' – vertical columns (robot sweeps up-down, columns advance left-right)
# For tall arenas use 'x'; for wide arenas use 'y'.
SWEEP_AXIS = 'x'

# Minimum clearance from any occupied cell.  Should be >= robot radius.
ROBOT_CLEARANCE  = 0.22     # metres

# Hard bounding box for the competition arena (map frame, metres).
# Waypoints outside this box are discarded, keeping the robot inside the
# fenced area and off the bridge.
#
# How to set these values:
#   1. Open RViz while the map is loaded.
#   2. Hover the mouse over each inner corner of the fence.
#   3. Read the (x, y) coordinates shown in the status bar at the bottom.
#   4. Fill in the min/max below.
#
# Set to None to disable the filter (not recommended – robot may leave arena).

# ARENA_X_MIN: float | None = -4.5
# ARENA_X_MAX: float | None =  3.0
# ARENA_Y_MIN: float | None = -1.0
# ARENA_Y_MAX: float | None =  8.0

ARENA_X_MIN: float | None =  None
ARENA_X_MAX: float | None =  None
ARENA_Y_MIN: float | None =  None
ARENA_Y_MAX: float | None =  None

NUM_FACES        = 6        # stop after greeting this many faces
APPROACH_DIST    = 0.7      # metres – stand this far from the face
GREETING_TEXT    = "Hello! I found your face. Pleased to meet you!"
ESPEAK_SPEED     = 140      # words per minute
LOOP_START_TEXT  = "Starting loop"


# ---------------------------------------------------------------------------

# QoS for the latched /map topic
_MAP_QOS = QoSProfile(
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1)


class State(Enum):
    SEARCHING   = auto()
    APPROACHING = auto()
    GREETING    = auto()
    DONE        = auto()


class Task1Node(RobotCommander):
    """Extends RobotCommander with boustrophedon search and face greeting."""

    def __init__(self):
        super().__init__(node_name='task1')

        # Coverage path – populated when the first /map message arrives
        self.coverage_waypoints: list[tuple[float, float, float]] = []

        # Face tracking
        self.known_faces:  dict[int, tuple[float, float, float]] = {}
        self.greeted_ids:  set[int]   = set()
        self.to_greet:     deque[int] = deque()

        # State machine
        self.state            = State.SEARCHING
        self.waypoint_idx     = 0
        self.current_face_id: int | None = None

        # Subscribe to the occupancy map (once) to build the coverage path
        self.create_subscription(
            OccupancyGrid, '/map', self._map_cb, _MAP_QOS)

        # Subscribe to confirmed face positions from detect_people.py
        self.create_subscription(
            MarkerArray, '/people_markers', self._marker_cb, 10)

        self.info('Task1 node ready – waiting for map and Nav2.')

    # ------------------------------------------------------------------
    # Map callback – build boustrophedon path (runs once)
    # ------------------------------------------------------------------

    def _map_cb(self, msg: OccupancyGrid) -> None:
        if self.coverage_waypoints:
            return      # already computed
        self.coverage_waypoints = self._boustrophedon(msg)
        self.info(
            f'Coverage path ready: {len(self.coverage_waypoints)} waypoints '
            f'(spacing={COVERAGE_SPACING} m).')

    def _boustrophedon(self, grid: OccupancyGrid) -> list[tuple[float, float, float]]:
        """Generate a lawnmower path over the free cells of *grid*.

        Returns a list of (world_x, world_y, yaw_deg) tuples.
        Headings are set so the robot faces toward the next waypoint,
        giving natural forward-facing motion in each row.
        """
        res   = grid.info.resolution
        gw    = grid.info.width
        gh    = grid.info.height
        ox    = grid.info.origin.position.x
        oy    = grid.info.origin.position.y

        # Occupancy values: 0=free, 100=occupied, -1=unknown
        data = np.array(grid.data, dtype=np.int8).reshape(gh, gw)

        step       = max(1, int(COVERAGE_SPACING / res))
        clearance  = max(1, int(ROBOT_CLEARANCE  / res))

        waypoints: list[tuple[float, float]] = []   # (world_x, world_y)

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
            # Vertical columns: outer loop advances left-right (ix),
            # inner loop sweeps up-down (iy), reversed every other column.
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
            # Horizontal rows: outer loop advances bottom-top (iy),
            # inner loop sweeps left-right (ix), reversed every other row.
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

        # Assign heading: each waypoint faces toward the next one
        result: list[tuple[float, float, float]] = []
        for i, (wx, wy) in enumerate(waypoints):
            if i < len(waypoints) - 1:
                nx, ny = waypoints[i + 1]
                yaw_deg = math.degrees(math.atan2(ny - wy, nx - wx))
            else:
                yaw_deg = result[-1][2] if result else 0.0
            result.append((wx, wy, yaw_deg))

        return result

    # ------------------------------------------------------------------
    # /people_markers callback
    # ------------------------------------------------------------------

    def _marker_cb(self, msg: MarkerArray) -> None:
        for m in msg.markers:
            if m.ns != 'faces':
                continue
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

    # ------------------------------------------------------------------
    # Navigation helpers
    # ------------------------------------------------------------------

    def _spin_ros(self, timeout: float = 0.1) -> None:
        rclpy.spin_once(self, timeout_sec=timeout)

    def _wait_nav(self, allow_interrupt: bool = True) -> bool:
        """Spin until current Nav2 goal finishes.

        Returns True on completion, False if interrupted by a queued face.
        """
        while not self.isTaskComplete():
            self._spin_ros()
            if allow_interrupt and self.to_greet:
                self.cancelTask()
                self.info('Navigation cancelled – new face queued.')
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
        """PoseStamped ~APPROACH_DIST m in front of face, facing it."""
        fx, fy, fz = self.known_faces[face_id]

        if hasattr(self, 'current_pose'):
            rx = self.current_pose.pose.position.x
            ry = self.current_pose.pose.position.y
        else:
            rx, ry = 0.0, 0.0

        dx, dy = rx - fx, ry - fy
        length = math.hypot(dx, dy)
        if length < 1e-3:
            dx, dy = 1.0, 0.0
        else:
            dx, dy = dx / length, dy / length

        ax  = fx + dx * APPROACH_DIST
        ay  = fy + dy * APPROACH_DIST
        yaw = math.atan2(fy - ay, fx - ax)

        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.header.stamp    = self.get_clock().now().to_msg()
        goal.pose.position.x = ax
        goal.pose.position.y = ay
        goal.pose.position.z = 0.0
        goal.pose.orientation = self.YawToQuaternion(yaw)
        return goal

    # ------------------------------------------------------------------
    # Greeting
    # ------------------------------------------------------------------

    def _greet(self) -> None:
        self.info(f'Speaking: "{GREETING_TEXT}"')
        proc = subprocess.Popen(
            ['espeak-ng', '-s', str(ESPEAK_SPEED), GREETING_TEXT],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL)
        while proc.poll() is None:
            self._spin_ros(timeout=0.05)

    def _say(self, text: str) -> None:
        proc = subprocess.Popen(
            ['espeak-ng', '-s', str(ESPEAK_SPEED), text],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL)
        while proc.poll() is None:
            self._spin_ros(timeout=0.05)

    # ------------------------------------------------------------------
    # Main state machine
    # ------------------------------------------------------------------

    def run(self) -> None:
        self.waitUntilNav2Active()

        # Wait for the boustrophedon path to be computed from /map
        self.info('Waiting for /map to build coverage path...')
        while not self.coverage_waypoints and rclpy.ok():
            self._spin_ros(0.1)

        # Rotate the path so the nearest waypoint to the robot's current
        # position is first – eliminates the long deadhead travel at startup.
        if hasattr(self, 'current_pose'):
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
        self._say(LOOP_START_TEXT)

        while rclpy.ok():

            # ── DONE ──────────────────────────────────────────────────
            if self.state == State.DONE:
                self.info(f'All {NUM_FACES} faces greeted.  Task complete!')
                break

            # ── SEARCHING ─────────────────────────────────────────────
            elif self.state == State.SEARCHING:

                if self.to_greet:
                    self.state = State.APPROACHING
                    continue

                if self.waypoint_idx >= len(self.coverage_waypoints):
                    if len(self.greeted_ids) >= NUM_FACES:
                        self.state = State.DONE
                    else:
                        self.info(
                            'Path exhausted; restarting '
                            f'({len(self.greeted_ids)}/{NUM_FACES} greeted).')
                        self._say(LOOP_START_TEXT)
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
                    self.state = State.APPROACHING
                else:
                    self.waypoint_idx += 1
                    if self.to_greet:
                        self.state = State.APPROACHING

            # ── APPROACHING ───────────────────────────────────────────
            elif self.state == State.APPROACHING:

                face_id = self.to_greet.popleft()
                if face_id in self.greeted_ids:
                    self.state = State.SEARCHING
                    continue

                self.current_face_id = face_id
                fx, fy, _ = self.known_faces[face_id]
                self.info(f'Approaching face #{face_id} at ({fx:.2f}, {fy:.2f}).')
                self.goToPose(self._approach_pose(face_id))
                self._wait_nav(allow_interrupt=False)
                self.state = State.GREETING

            # ── GREETING ──────────────────────────────────────────────
            elif self.state == State.GREETING:

                face_id = self.current_face_id
                self._greet()
                self.greeted_ids.add(face_id)
                self.info(
                    f'Greeted face #{face_id}.  '
                    f'Total: {len(self.greeted_ids)}/{NUM_FACES}.')

                self.state = (State.DONE
                              if len(self.greeted_ids) >= NUM_FACES
                              else State.SEARCHING)

            self._spin_ros()


# ---------------------------------------------------------------------------

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
