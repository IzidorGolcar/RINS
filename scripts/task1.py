#!/usr/bin/env python3
import math
import os
import subprocess
import sys
from collections import deque
from enum import Enum, auto

import random
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
from rclpy.qos import (QoSDurabilityPolicy, QoSHistoryPolicy,
                        QoSProfile, QoSReliabilityPolicy)
from visualization_msgs.msg import MarkerArray

sys.path.insert(0, os.path.dirname(__file__))
from robot_commander import RobotCommander  # noqa: E402


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

ARENA_X_MIN: float | None =  None
ARENA_X_MAX: float | None =  None
ARENA_Y_MIN: float | None =  None
ARENA_Y_MAX: float | None =  None

NUM_FACES        = 3
NUM_RINGS        = 2
APPROACH_DIST    = 0.7      # metres – stand this far from the face / ring
GREETING_TEXT    = "Hello! I found your face. Pleased to meet you!"
RING_GREETING_TEMPLATE = "Hello! I found a {color} ring!"
ESPEAK_SPEED     = 140      # words per minute


def _bgr_to_color_name(bgr: tuple[int, int, int]) -> str:
    """Map an average BGR colour to a human-readable name."""
    b, g, r = int(bgr[0]), int(bgr[1]), int(bgr[2])
    mx = max(r, g, b)
    mn = min(r, g, b)
    sat = (mx - mn) / max(mx, 1)

    if sat < 0.20:
        if mx > 200:
            return 'white'
        if mx < 60:
            return 'black'
        return 'grey'

    # Hue in [0, 360)
    if mx == r:
        h = 60.0 * ((g - b) / (mx - mn) % 6)
    elif mx == g:
        h = 60.0 * ((b - r) / (mx - mn) + 2)
    else:
        h = 60.0 * ((r - g) / (mx - mn) + 4)

    if h < 15 or h >= 345:
        return 'red'
    if h < 45:
        return 'orange'
    if h < 75:
        return 'yellow'
    if h < 150:
        return 'green'
    if h < 195:
        return 'cyan'
    if h < 255:
        return 'blue'
    if h < 315:
        return 'purple'
    return 'pink'



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

        self.coverage_waypoints: list[tuple[float, float, float]] = []

        self.known_faces:  dict[int, tuple[float, float, float]] = {}
        self.greeted_ids:  set[int]   = set()
        self.to_greet:     deque[int] = deque()

        self.known_rings:       dict[int, tuple[float, float, float]] = {}
        self.ring_colors:       dict[int, tuple[int, int, int]]       = {}  # rid -> BGR
        self.greeted_ring_ids:  set[int]   = set()
        self.to_greet_rings:    deque[int] = deque()

        self.state            = State.SEARCHING
        self.waypoint_idx     = 0
        self.current_face_id: int | None = None
        self.current_ring_id: int | None = None

        self.create_subscription(
            OccupancyGrid, '/map', self._map_cb, _MAP_QOS)

        self.create_subscription(
            MarkerArray, '/people_markers', self._people_marker_cb, 10)

        self.create_subscription(
            MarkerArray, '/ring_markers', self._ring_marker_cb, 10)

        self.info('Task1 node ready – waiting for map and Nav2.')

    def _map_cb(self, msg: OccupancyGrid) -> None:
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

        data = np.array(grid.data, dtype=np.int8).reshape(gh, gw)

        step       = max(1, int(COVERAGE_SPACING / res))
        clearance  = max(1, int(ROBOT_CLEARANCE  / res))

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


    def _people_marker_cb(self, msg: MarkerArray) -> None:
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
                    self.to_greet_rings.append(rid)
                    color_name = _bgr_to_color_name(bgr)
                    self.info(
                        f'New {color_name} ring #{rid} queued at '
                        f'({pos[0]:.2f}, {pos[1]:.2f})')


    def _spin_ros(self, timeout: float = 0.1) -> None:
        rclpy.spin_once(self, timeout_sec=timeout)

    def _wait_nav(self, allow_interrupt: bool = True) -> bool:
        """Spin until current Nav2 goal finishes.

        Returns True on successful completion, False if interrupted or failed.
        """
        while not self.isTaskComplete():
            self._spin_ros()
            if allow_interrupt and (self.to_greet or self.to_greet_rings):
                self.cancelTask()
                self.info('Navigation cancelled – new target queued.')
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
        """PoseStamped ~APPROACH_DIST m in front of face, facing it."""
        fx, fy, fz = self.known_faces[face_id]
        return self._approach_pose_for(fx, fy, fz)

    def _approach_pose_ring(self, ring_id: int) -> PoseStamped:
        """PoseStamped ~APPROACH_DIST m in front of ring, facing it."""
        rx2, ry2, rz2 = self.known_rings[ring_id]
        return self._approach_pose_for(rx2, ry2, rz2)

    def _approach_pose_for(self, fx: float, fy: float, fz: float) -> PoseStamped:
        """Compute approach PoseStamped for any (fx, fy, fz) target."""
        # FIX: use current_pose if available, but log a warning if it isn't,
        # since falling back to (0, 0) produces a bogus approach direction.
        if hasattr(self, 'current_pose') and self.current_pose is not None:
            rx = self.current_pose.pose.position.x
            ry = self.current_pose.pose.position.y
        else:
            self.warn(
                'current_pose not available – approach direction may be wrong. '
                'Ensure RobotCommander subscribes to a pose topic.')
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


    def run(self) -> None:
        self.waitUntilNav2Active()

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
        self._say("")

        while rclpy.ok():

            if self.state == State.DONE:
                self.info(
                    f'All {NUM_FACES} faces and {NUM_RINGS} rings greeted. '
                    f'Task complete!')
                self._say("I'm done! That was easy!")
                break

            elif self.state == State.SEARCHING:

                if self.to_greet or self.to_greet_rings:
                    self.state = State.APPROACHING
                    continue

                if self.waypoint_idx >= len(self.coverage_waypoints):
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
                    self.state = State.APPROACHING
                else:
                    self.waypoint_idx += 1
                    if self.to_greet or self.to_greet_rings:
                        self.state = State.APPROACHING

            elif self.state == State.APPROACHING:

                if self.to_greet:
                    face_id = self.to_greet.popleft()
                    if face_id in self.greeted_ids:
                        self.state = State.SEARCHING
                        continue

                    self.current_face_id = face_id
                    self.current_ring_id = None
                    fx, fy, _ = self.known_faces[face_id]
                    self.info(
                        f'Approaching face #{face_id} at ({fx:.2f}, {fy:.2f}).')
                    self.goToPose(self._approach_pose(face_id))

                    # FIX: check whether approach navigation actually succeeded
                    # before transitioning to GREETING. If it failed, re-queue
                    # the face and go back to searching rather than greeting
                    # from wherever the robot currently is.
                    completed = self._wait_nav(allow_interrupt=False)
                    if completed:
                        self.state = State.GREETING
                    else:
                        self.warn(
                            f'Failed to reach approach pose for face #{face_id}. '
                            f'Re-queuing.')
                        self.to_greet.appendleft(face_id)
                        self.current_face_id = None
                        self.state = State.SEARCHING

                elif self.to_greet_rings:
                    ring_id = self.to_greet_rings.popleft()
                    if ring_id in self.greeted_ring_ids:
                        self.state = State.SEARCHING
                        continue

                    self.current_ring_id = ring_id
                    self.current_face_id = None
                    rx2, ry2, _ = self.known_rings[ring_id]
                    color_name = _bgr_to_color_name(self.ring_colors[ring_id])
                    self.info(
                        f'Approaching {color_name} ring #{ring_id} '
                        f'at ({rx2:.2f}, {ry2:.2f}).')
                    self.goToPose(self._approach_pose_ring(ring_id))

                    # FIX: same fix as for faces – only greet after confirmed arrival
                    completed = self._wait_nav(allow_interrupt=False)
                    if completed:
                        self.state = State.GREETING
                    else:
                        self.warn(
                            f'Failed to reach approach pose for ring #{ring_id}. '
                            f'Re-queuing.')
                        self.to_greet_rings.appendleft(ring_id)
                        self.current_ring_id = None
                        self.state = State.SEARCHING

                else:
                    self.state = State.SEARCHING

            elif self.state == State.GREETING:

                if self.current_face_id is not None:
                    face_id = self.current_face_id
                    self._greet()
                    self.greeted_ids.add(face_id)
                    self.info(
                        f'Greeted face #{face_id}.  '
                        f'Total faces: {len(self.greeted_ids)}/{NUM_FACES}.')

                elif self.current_ring_id is not None:
                    ring_id = self.current_ring_id
                    color_name = _bgr_to_color_name(self.ring_colors[ring_id])
                    text = RING_GREETING_TEMPLATE.format(color=color_name)
                    self._say(text)
                    self.greeted_ring_ids.add(ring_id)
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