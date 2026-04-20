#!/usr/bin/env python3
#
# Real-robot bring-up: localization + Nav2 + RViz.
# Gazebo and robot spawning are intentionally removed; this launch file is
# for a physical TurtleBot4 whose drivers are already running on the robot.

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node

pkg_dis_tutorial3 = get_package_share_directory('dis_tutorial3')
pkg_turtlebot4_viz = get_package_share_directory('turtlebot4_viz')

ARGUMENTS = [
    DeclareLaunchArgument('namespace', default_value='',
                          description='Robot namespace'),
    DeclareLaunchArgument('rviz', default_value='true', choices=['true', 'false'],
                          description='Start rviz.'),
    DeclareLaunchArgument('use_sim_time', default_value='false', choices=['true', 'false'],
                          description='use_sim_time (must be false on real robot).'),
    DeclareLaunchArgument(
        'map',
        default_value=PathJoinSubstitution([pkg_dis_tutorial3, 'maps', 'task1r.yaml']),
        description='Full path to map yaml file to load'),
]


def generate_launch_description():
    localization_launch = PathJoinSubstitution(
        [pkg_dis_tutorial3, 'launch', 'localization.launch.py'])
    nav2_launch = PathJoinSubstitution(
        [pkg_dis_tutorial3, 'launch', 'nav2.launch.py'])
    rviz_launch = PathJoinSubstitution(
        [pkg_turtlebot4_viz, 'launch', 'view_navigation.launch.py'])

    localization = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([localization_launch]),
        launch_arguments=[
            ('namespace', LaunchConfiguration('namespace')),
            ('use_sim_time', LaunchConfiguration('use_sim_time')),
            ('map', LaunchConfiguration('map')),
        ],
    )

    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([nav2_launch]),
        launch_arguments=[
            ('namespace', LaunchConfiguration('namespace')),
            ('use_sim_time', LaunchConfiguration('use_sim_time')),
        ],
    )

    rviz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([rviz_launch]),
        launch_arguments=[
            ('namespace', LaunchConfiguration('namespace')),
        ],
        condition=IfCondition(LaunchConfiguration('rviz')),
    )

    auto_localize = Node(
        package='dis_tutorial3',
        executable='auto_localize.py',
        name='auto_localize',
        output='screen',
    )

    ld = LaunchDescription(ARGUMENTS)
    ld.add_action(localization)
    ld.add_action(nav2)
    ld.add_action(rviz)
    ld.add_action(auto_localize)
    return ld
