#!/usr/bin/env python3

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node

PKG = 'dis_tutorial3'

ARGUMENTS = [
    DeclareLaunchArgument(
        'map',
        default_value=PathJoinSubstitution(
            [get_package_share_directory(PKG), 'maps', 'task1r.yaml']
        ),
        description='Full path to map yaml file',
    ),
    DeclareLaunchArgument(
        'namespace', default_value='', description='Robot namespace'
    ),
    # Gemini 355L camera topics – override if your driver uses different names
    DeclareLaunchArgument('rgb_topic',    default_value='/gemini/color/image_raw'),
    DeclareLaunchArgument('depth_topic',  default_value='/gemini/depth/image_raw'),
    DeclareLaunchArgument('points_topic', default_value='/gemini/depth_registered/points'),
    DeclareLaunchArgument('camera_frame', default_value='gemini_color_frame'),
]


def generate_launch_description():
    pkg = get_package_share_directory(PKG)

    localization = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg, 'launch', 'localization.launch.py'])
        ),
        launch_arguments={
            'namespace': LaunchConfiguration('namespace'),
            'map':       LaunchConfiguration('map'),
            'use_sim_time': 'false',
        }.items(),
    )

    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg, 'launch', 'nav2.launch.py'])
        ),
        launch_arguments={
            'namespace':    LaunchConfiguration('namespace'),
            'use_sim_time': 'false',
        }.items(),
    )

    detect_people = Node(
        package=PKG,
        executable='detect_people.py',
        name='face_detector',
        output='screen',
        parameters=[{
            'rgb_topic':   LaunchConfiguration('rgb_topic'),
            'depth_topic': LaunchConfiguration('points_topic'),
        }],
    )

    detect_rings = Node(
        package=PKG,
        executable='detect_rings.py',
        name='ring_detector',
        output='screen',
        parameters=[{
            'rgb_topic':    LaunchConfiguration('rgb_topic'),
            'depth_topic':  LaunchConfiguration('depth_topic'),
            'camera_frame': LaunchConfiguration('camera_frame'),
        }],
    )

    task1 = Node(
        package=PKG,
        executable='task1.py',
        name='task1',
        output='screen',
    )

    ld = LaunchDescription(ARGUMENTS)
    ld.add_action(localization)
    ld.add_action(nav2)
    ld.add_action(detect_people)
    ld.add_action(detect_rings)
    ld.add_action(task1)
    return ld
