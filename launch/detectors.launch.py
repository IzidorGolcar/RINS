from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pkg = 'dis_tutorial3'

    return LaunchDescription([
        Node(package=pkg, executable='detect_rings.py',
             name='detect_rings', output='screen'),

        Node(package=pkg, executable='detect_people.py',
             name='detect_people', output='screen'),

        Node(package=pkg, executable='detect_barrels.py',
             name='detect_barrels', output='screen'),

        Node(package=pkg, executable='line_detection.py',
             name='line_detection', output='screen'),

        Node(package=pkg, executable='yellow_line_obstacle.py',
             name='yellow_line_obstacle', output='screen'),

        Node(package=pkg, executable='anomaly_detector.py',
             name='anomaly_detector', output='screen'),

        Node(package=pkg, executable='qr_detector.py',
             name='qr_detector', output='screen'),

        Node(package=pkg, executable='line_follower.py',
             name='line_follower', output='screen'),

        # Required for QR fallback in dialogue: arm_mover_actions listens on
        # /arm_command and parks the wrist camera at the 'look_for_qr' pose;
        # dialogue_node publishes the prompt + decodes the QR frame.
        Node(package=pkg, executable='arm_mover_actions.py',
             name='arm_mover_actions', output='screen'),

        Node(package=pkg, executable='dialogue_node.py',
             name='dialogue_node', output='screen'),
    ])
