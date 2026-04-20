#!/usr/bin/env python3
"""One-shot global localization helper.

Called at start-up from sim_turtlebot_nav.launch.py. It:
  1. Waits for AMCL and the Nav2 spin action to come up.
  2. Asks AMCL to scatter its particles over the whole map
     (`reinitialize_global_localization`).
  3. Commands two full rotations so laser data lets the filter converge.
  4. Exits.

After this node exits, the robot is correctly placed on the map in RViz,
so task1 can start without any manual initial-pose step.
"""

import math
import time

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from builtin_interfaces.msg import Duration
from lifecycle_msgs.srv import GetState
from nav2_msgs.action import Spin
from std_srvs.srv import Empty as EmptySrv


class AutoLocalize(Node):
    def __init__(self):
        super().__init__('auto_localize')
        self._spin_client = ActionClient(self, Spin, 'spin')

    def _wait_node_active(self, node_name: str) -> None:
        client = self.create_client(GetState, f'{node_name}/get_state')
        while not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f'Waiting for {node_name}/get_state...')
        while rclpy.ok():
            future = client.call_async(GetState.Request())
            rclpy.spin_until_future_complete(self, future)
            if future.result() and future.result().current_state.label == 'active':
                return
            time.sleep(1.0)

    def _reinitialize_amcl(self) -> bool:
        client = self.create_client(EmptySrv, 'reinitialize_global_localization')
        if not client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error(
                'reinitialize_global_localization service not available.')
            return False
        client.call_async(EmptySrv.Request())
        self.get_logger().info('AMCL particles scattered over the map.')
        return True

    def _spin_full(self, turns: int = 2) -> None:
        self.get_logger().info('Waiting for Nav2 spin action server...')
        while not self._spin_client.wait_for_server(timeout_sec=1.0):
            pass
        for i in range(turns):
            goal = Spin.Goal()
            goal.target_yaw = math.pi * 2.1
            goal.time_allowance = Duration(sec=20)
            self.get_logger().info(f'Spin {i + 1}/{turns}...')
            send_future = self._spin_client.send_goal_async(goal)
            rclpy.spin_until_future_complete(self, send_future)
            handle = send_future.result()
            if handle is None or not handle.accepted:
                self.get_logger().warn('Spin goal rejected.')
                return
            result_future = handle.get_result_async()
            rclpy.spin_until_future_complete(self, result_future)

    def run(self) -> None:
        self._wait_node_active('amcl')
        self._wait_node_active('bt_navigator')
        time.sleep(1.5)
        if not self._reinitialize_amcl():
            return
        time.sleep(1.0)
        self._spin_full(turns=2)
        self.get_logger().info('Localization complete – robot pose is valid.')


def main() -> None:
    rclpy.init(args=None)
    node = AutoLocalize()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
