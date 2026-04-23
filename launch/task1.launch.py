from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():

    dis_nav2_params = PathJoinSubstitution([
        FindPackageShare('dis_tutorial3'), 'config', 'nav2.yaml'
    ])

    # Localization launch — keep TurtleBot4's default localization.yaml
    # (our config/localization.yaml uses scan_filtered, which would require
    # the laser_filter_chain to be running and would break 2D Pose Estimate
    # in RViz without it).
    localization_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('turtlebot4_navigation'),
                'launch',
                'localization.launch.py'
            ])
        ),
        launch_arguments={
            'map': 'src/RINS/maps/task1r.yaml',
        }.items()
    )

    # Navigation launch
    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('turtlebot4_navigation'),
                'launch',
                'nav2.launch.py'
            ])
        ),
        launch_arguments={
            'params_file': dis_nav2_params,
        }.items()
    )

    # Visualization launch
    viz_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('turtlebot4_viz'),
                'launch',
                'view_navigation.launch.py'
            ])
        )
    )

    return LaunchDescription([
        localization_launch,
        nav2_launch,
        viz_launch
    ])