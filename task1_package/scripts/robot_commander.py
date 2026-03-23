#! /usr/bin/env python3
# Mofidied from Samsung Research America
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from enum import Enum
import math
import time

from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import PointStamped, Quaternion, PoseStamped, PoseWithCovarianceStamped
from lifecycle_msgs.srv import GetState
from nav2_msgs.action import Spin, NavigateToPose
from std_msgs.msg import String
from turtle_tf2_py.turtle_tf2_broadcaster import quaternion_from_euler

from irobot_create_msgs.action import Dock, Undock
from irobot_create_msgs.msg import DockStatus

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from rclpy.qos import qos_profile_sensor_data


class TaskResult(Enum):
    UNKNOWN = 0
    SUCCEEDED = 1
    CANCELED = 2
    FAILED = 3

amcl_pose_qos = QoSProfile(
          durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
          reliability=QoSReliabilityPolicy.RELIABLE,
          history=QoSHistoryPolicy.KEEP_LAST,
          depth=1)

class RobotCommander(Node):

    def __init__(self, node_name='robot_commander', namespace=''):
        super().__init__(node_name=node_name, namespace=namespace)
        
        self.pose_frame_id = 'map'
        
        self.goal_handle = None
        self.result_future = None
        self.feedback = None
        self.status = None
        self.initial_pose_received = False
        self.is_docked = None
        self.current_pose = None

        self.declare_parameters(
            namespace='',
            parameters=[
                ('required_faces', 3),
                ('required_rings', 2),
                ('max_mission_seconds', 480.0),
                ('max_search_rounds', 2),
                ('search_spin_radians', 3.14),
                ('search_waypoints', '0.0,0.0;1.6,0.0;1.6,1.3;0.0,1.3;-1.2,1.0;-1.2,-0.8;0.8,-1.2'),
            ],
        )

        self.required_faces = int(self.get_parameter('required_faces').value)
        self.required_rings = int(self.get_parameter('required_rings').value)
        self.max_mission_seconds = float(self.get_parameter('max_mission_seconds').value)
        self.max_search_rounds = int(self.get_parameter('max_search_rounds').value)
        self.search_spin_radians = float(self.get_parameter('search_spin_radians').value)

        self.search_waypoints = self._parse_waypoints(str(self.get_parameter('search_waypoints').value))

        self.detected_faces = []
        self.detected_rings = []

        self.create_subscription(DockStatus, 'dock_status', self._dockCallback, qos_profile_sensor_data)
        self.localization_pose_sub = self.create_subscription(PoseWithCovarianceStamped, 'amcl_pose', self._amclPoseCallback, amcl_pose_qos)
        self.create_subscription(PointStamped, '/detected_faces', self._faceDetectedCallback, 10)
        self.create_subscription(String, '/detected_rings_color', self._ringDetectedCallback, 10)
        
        self.initial_pose_pub = self.create_publisher(PoseWithCovarianceStamped, 'initialpose', 10)
        self.speech_pub = self.create_publisher(String, '/speech_event', 10)
        
        self.nav_to_pose_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.spin_client = ActionClient(self, Spin, 'spin')
        self.undock_action_client = ActionClient(self, Undock, 'undock')
        self.dock_action_client = ActionClient(self, Dock, 'dock')

        self.get_logger().info(f"Robot commander has been initialized!")

    def play_speech(self, key, fallback_text=''):
        msg = String()
        msg.data = key
        self.speech_pub.publish(msg)

        if fallback_text:
            self.info(fallback_text)
        
    def destroyNode(self):
        self.nav_to_pose_client.destroy()
        super().destroy_node()     

    def goToPose(self, pose, behavior_tree=''):
        """Send a `NavToPose` action request."""
        self.debug("Waiting for 'NavigateToPose' action server")
        while not self.nav_to_pose_client.wait_for_server(timeout_sec=1.0):
            self.info("'NavigateToPose' action server not available, waiting...")

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose
        goal_msg.behavior_tree = behavior_tree

        self.info('Navigating to goal: ' + str(pose.pose.position.x) + ' ' +
                  str(pose.pose.position.y) + '...')
        send_goal_future = self.nav_to_pose_client.send_goal_async(goal_msg,
                                                                   self._feedbackCallback)
        rclpy.spin_until_future_complete(self, send_goal_future)
        self.goal_handle = send_goal_future.result()

        if not self.goal_handle.accepted:
            self.error('Goal to ' + str(pose.pose.position.x) + ' ' +
                       str(pose.pose.position.y) + ' was rejected!')
            return False

        self.result_future = self.goal_handle.get_result_async()
        return True

    def spin(self, spin_dist=1.57, time_allowance=10):
        self.debug("Waiting for 'Spin' action server")
        while not self.spin_client.wait_for_server(timeout_sec=1.0):
            self.info("'Spin' action server not available, waiting...")
        goal_msg = Spin.Goal()
        goal_msg.target_yaw = spin_dist
        goal_msg.time_allowance = Duration(sec=time_allowance)

        self.info(f'Spinning to angle {goal_msg.target_yaw}....')
        send_goal_future = self.spin_client.send_goal_async(goal_msg, self._feedbackCallback)
        rclpy.spin_until_future_complete(self, send_goal_future)
        self.goal_handle = send_goal_future.result()

        if not self.goal_handle.accepted:
            self.error('Spin request was rejected!')
            return False

        self.result_future = self.goal_handle.get_result_async()
        return True
    
    def undock(self):
        """Perform Undock action."""
        self.info('Undocking...')
        self.undock_send_goal()

        while not self.isUndockComplete():
            time.sleep(0.1)

    def undock_send_goal(self):
        goal_msg = Undock.Goal()
        self.undock_action_client.wait_for_server()
        goal_future = self.undock_action_client.send_goal_async(goal_msg)

        rclpy.spin_until_future_complete(self, goal_future)

        self.undock_goal_handle = goal_future.result()

        if not self.undock_goal_handle.accepted:
            self.error('Undock goal rejected')
            return

        self.undock_result_future = self.undock_goal_handle.get_result_async()

    def isUndockComplete(self):
        """
        Get status of Undock action.

        :return: ``True`` if undocked, ``False`` otherwise.
        """
        if self.undock_result_future is None or not self.undock_result_future:
            return True

        rclpy.spin_until_future_complete(self, self.undock_result_future, timeout_sec=0.1)

        if self.undock_result_future.result():
            self.undock_status = self.undock_result_future.result().status
            if self.undock_status != GoalStatus.STATUS_SUCCEEDED:
                self.info(f'Goal with failed with status code: {self.status}')
                return True
        else:
            return False

        self.info('Undock succeeded')
        return True

    def cancelTask(self):
        """Cancel pending task request of any type."""
        self.info('Canceling current task.')
        if self.result_future:
            future = self.goal_handle.cancel_goal_async()
            rclpy.spin_until_future_complete(self, future)
        return

    def isTaskComplete(self):
        """Check if the task request of any type is complete yet."""
        if not self.result_future:
            return True
        rclpy.spin_until_future_complete(self, self.result_future, timeout_sec=0.10)
        if self.result_future.result():
            self.status = self.result_future.result().status
            if self.status != GoalStatus.STATUS_SUCCEEDED:
                self.debug(f'Task with failed with status code: {self.status}')
                return True
        else:
            return False

        self.debug('Task succeeded!')
        return True

    def getFeedback(self):
        """Get the pending action feedback message."""
        return self.feedback

    def getResult(self):
        """Get the pending action result message."""
        if self.status == GoalStatus.STATUS_SUCCEEDED:
            return TaskResult.SUCCEEDED
        elif self.status == GoalStatus.STATUS_ABORTED:
            return TaskResult.FAILED
        elif self.status == GoalStatus.STATUS_CANCELED:
            return TaskResult.CANCELED
        else:
            return TaskResult.UNKNOWN

    def waitUntilNav2Active(self, navigator='bt_navigator', localizer='amcl'):
        """Block until the full navigation system is up and running."""
        self._waitForNodeToActivate(localizer)
        if not self.initial_pose_received:
            time.sleep(1)
        self._waitForNodeToActivate(navigator)
        self.info('Nav2 is ready for use!')
        return

    def _waitForNodeToActivate(self, node_name):
        # Waits for the node within the tester namespace to become active
        self.debug(f'Waiting for {node_name} to become active..')
        node_service = f'{node_name}/get_state'
        state_client = self.create_client(GetState, node_service)
        while not state_client.wait_for_service(timeout_sec=1.0):
            self.info(f'{node_service} service not available, waiting...')

        req = GetState.Request()
        state = 'unknown'
        while state != 'active':
            self.debug(f'Getting {node_name} state...')
            future = state_client.call_async(req)
            rclpy.spin_until_future_complete(self, future)
            if future.result() is not None:
                state = future.result().current_state.label
                self.debug(f'Result of get_state: {state}')
            time.sleep(2)
        return
    
    def YawToQuaternion(self, angle_z = 0.):
        quat_tf = quaternion_from_euler(0, 0, angle_z)

        # Convert a list to geometry_msgs.msg.Quaternion
        quat_msg = Quaternion(x=quat_tf[0], y=quat_tf[1], z=quat_tf[2], w=quat_tf[3])
        return quat_msg

    def _amclPoseCallback(self, msg):
        self.debug('Received amcl pose')
        self.initial_pose_received = True
        self.current_pose = msg.pose
        return

    def _feedbackCallback(self, msg):
        self.debug('Received action feedback message')
        self.feedback = msg.feedback
        return
    
    def _dockCallback(self, msg: DockStatus):
        self.is_docked = msg.is_docked

    def setInitialPose(self, pose):
        msg = PoseWithCovarianceStamped()
        msg.pose.pose = pose
        msg.header.frame_id = self.pose_frame_id
        msg.header.stamp = 0
        self.info('Publishing Initial Pose')
        self.initial_pose_pub.publish(msg)
        return

    def info(self, msg):
        self.get_logger().info(msg)
        return

    def warn(self, msg):
        self.get_logger().warn(msg)
        return

    def error(self, msg):
        self.get_logger().error(msg)
        return

    def make_face_id(self, x, y):
        return f"face:{round(x, 1)}:{round(y, 1)}"

    def make_ring_id(self, x, y, color):
        return f"ring:{color}:{round(x, 1)}:{round(y, 1)}"

    def gather_targets(self, handled_ids):
        targets = []

        for fx, fy, fz in self.detected_faces:
            face_id = self.make_face_id(fx, fy)
            if face_id in handled_ids:
                continue
            targets.append({'id': face_id, 'type': 'face', 'x': fx, 'y': fy, 'z': fz, 'label': 'face'})

        for ring in self.detected_rings:
            ring_id = self.make_ring_id(ring['x'], ring['y'], ring['color'])
            if ring_id in handled_ids:
                continue
            targets.append({
                'id': ring_id,
                'type': 'ring',
                'x': ring['x'],
                'y': ring['y'],
                'z': ring['z'],
                'label': ring['color'],
            })

        return targets

    def _distance_from_robot(self, x, y):
        if self.current_pose is None:
            return 0.0
        dx = x - self.current_pose.pose.position.x
        dy = y - self.current_pose.pose.position.y
        return math.hypot(dx, dy)

    def _goal_from_xy(self, x, y):
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        goal_pose.pose.position.x = x
        goal_pose.pose.position.y = y
        goal_pose.pose.orientation = self.YawToQuaternion(0.0)
        return goal_pose

    def _faceDetectedCallback(self, msg: PointStamped):
        if msg.header.frame_id != 'map':
            return
        face_id = self.make_face_id(msg.point.x, msg.point.y)
        for fx, fy, fz in self.detected_faces:
            if self.make_face_id(fx, fy) == face_id:
                return
        self.detected_faces.append((float(msg.point.x), float(msg.point.y), float(msg.point.z)))

    def _ringDetectedCallback(self, msg: String):
        payload = msg.data.strip()
        if ':' not in payload:
            return
        color, coords = payload.split(':', 1)
        if ',' not in coords:
            return
        parts = coords.split(',')
        if len(parts) != 3:
            return
        try:
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
        except ValueError:
            return

        ring = {'x': x, 'y': y, 'z': z, 'color': color.lower()}
        ring_id = self.make_ring_id(ring['x'], ring['y'], ring['color'])
        for existing in self.detected_rings:
            if self.make_ring_id(existing['x'], existing['y'], existing['color']) == ring_id:
                return
        self.detected_rings.append(ring)

    def spin_some(self, duration=0.05):
        rclpy.spin_once(self, timeout_sec=duration)

    def _parse_waypoints(self, text):
        waypoints = []
        for chunk in text.split(';'):
            item = chunk.strip()
            if not item:
                continue
            coords = item.split(',')
            if len(coords) != 2:
                continue
            try:
                waypoints.append((float(coords[0]), float(coords[1])))
            except ValueError:
                continue
        return waypoints

    def debug(self, msg):
        self.get_logger().debug(msg)
        return
    
def main(args=None):

    rclpy.init(args=args)
    rc = RobotCommander()

    rc.waitUntilNav2Active()

    while rc.is_docked is None:
        rclpy.spin_once(rc, timeout_sec=0.5)

    if rc.is_docked:
        rc.undock()

    # Allow subscriptions to collect initial detections before mission loop starts.
    for _ in range(20):
        rc.spin_some(0.05)

    if not rc.search_waypoints:
        rc.warn('No search waypoints configured; robot will only react to already detected targets.')

    mission_start = time.time()
    handled_ids = set()
    face_done = 0
    ring_done = 0
    waypoint_cursor = 0
    search_round = 0

    rc.info(
        f'Starting autonomous mission. Need {rc.required_faces} faces and {rc.required_rings} rings.'
    )

    while True:
        if face_done >= rc.required_faces and ring_done >= rc.required_rings:
            rc.info('All required faces and rings completed. Stopping mission.')
            rc.play_speech('mission_done', 'Mission complete.')
            break

        elapsed = time.time() - mission_start
        if elapsed > rc.max_mission_seconds:
            rc.warn(f'Mission timeout reached after {elapsed:.1f}s. Stopping.')
            break

        rc.spin_some(0.05)

        targets = rc.gather_targets(handled_ids)
        target_candidates = []
        for target in targets:
            if target['type'] == 'face' and face_done >= rc.required_faces:
                continue
            if target['type'] == 'ring' and ring_done >= rc.required_rings:
                continue
            target_candidates.append((rc._distance_from_robot(target['x'], target['y']), target))

        if target_candidates:
            target_candidates.sort(key=lambda item: item[0])
            target = target_candidates[0][1]

            rc.info(f"Approaching detected {target['type']} at ({target['x']:.2f}, {target['y']:.2f}).")
            goal_pose = rc._goal_from_xy(target['x'], target['y'])
            if not rc.goToPose(goal_pose):
                rc.warn('Goal was rejected, continuing mission.')
                continue

            while not rc.isTaskComplete():
                time.sleep(0.5)

            result = rc.getResult()
            if result != TaskResult.SUCCEEDED:
                rc.warn(f"Failed to reach {target['type']} target.")
                continue

            handled_ids.add(target['id'])

            if target['type'] == 'face':
                face_done += 1
                rc.play_speech('greet', 'Hello face!')
                rc.info(f'Face completed ({face_done}/{rc.required_faces}).')
            else:
                ring_done += 1
                ring_color = target['label']
                rc.play_speech(ring_color, f'Ring color is {ring_color}.')
                rc.info(f'Ring completed ({ring_done}/{rc.required_rings}): {ring_color}')

            rc.spin(1.0)
            while not rc.isTaskComplete():
                time.sleep(0.2)
            continue

        if not rc.search_waypoints:
            rc.warn('No pending targets and no waypoints left for search. Stopping.')
            break

        wp_x, wp_y = rc.search_waypoints[waypoint_cursor]
        waypoint_cursor += 1
        if waypoint_cursor >= len(rc.search_waypoints):
            waypoint_cursor = 0
            search_round += 1
            if search_round >= rc.max_search_rounds:
                rc.warn('Reached maximum search rounds without finding all targets. Stopping.')
                break

        rc.info(f'Patrolling waypoint ({wp_x:.2f}, {wp_y:.2f}) for search.')
        waypoint_goal = rc._goal_from_xy(wp_x, wp_y)
        if rc.goToPose(waypoint_goal):
            while not rc.isTaskComplete():
                rc.spin_some(0.05)
                time.sleep(0.5)

            if rc.getResult() == TaskResult.SUCCEEDED:
                rc.spin(rc.search_spin_radians)
                while not rc.isTaskComplete():
                    time.sleep(0.2)

    rc.info(f'Mission ended. Faces: {face_done}/{rc.required_faces}, rings: {ring_done}/{rc.required_rings}.')
    rc.destroyNode()
if __name__=="__main__":
    main()