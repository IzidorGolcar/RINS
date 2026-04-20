#!/usr/bin/env python3

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution

PKG = 'dis_tutorial3'


def generate_launch_description():
    pkg     = get_package_share_directory(PKG)
    pkg_viz = get_package_share_directory('turtlebot4_viz')

    localization = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg, 'launch', 'localization.launch.py'])
        ),
        launch_arguments={
            'map':          PathJoinSubstitution([pkg, 'maps', 'task1r.yaml']),
            'use_sim_time': 'false',
        }.items(),
    )

    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg, 'launch', 'nav2.launch.py'])
        ),
        launch_arguments={
            'use_sim_time': 'false',
        }.items(),
    )

    rviz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_viz, 'launch', 'view_navigation.launch.py'])
        ),
    )

    ld = LaunchDescription()
    ld.add_action(localization)
    ld.add_action(nav2)
    ld.add_action(rviz)
    return ld
