from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from ament_index_python.packages import get_package_share_directory


ARGUMENTS = [
    DeclareLaunchArgument('namespace', default_value='', description='Robot namespace'),
    DeclareLaunchArgument('rviz', default_value='true', choices=['true', 'false'], description='Start rviz'),
    DeclareLaunchArgument('world', default_value='task2', description='Simulation world'),
    DeclareLaunchArgument('model', default_value='standard', choices=['standard', 'lite'], description='Turtlebot4 model'),
    DeclareLaunchArgument('use_sim_time', default_value='true', choices=['true', 'false'], description='Use sim time'),
    DeclareLaunchArgument(
        'map',
        default_value=PathJoinSubstitution([get_package_share_directory('task1_package'), 'maps', 'task1.yaml']),
        description='Map yaml path for localization',
    ),
    DeclareLaunchArgument('required_faces', default_value='3'),
    DeclareLaunchArgument('required_rings', default_value='2'),
    DeclareLaunchArgument('max_mission_seconds', default_value='480.0'),
    DeclareLaunchArgument('max_search_rounds', default_value='2'),
    DeclareLaunchArgument(
        'search_waypoints',
        default_value='0.0,0.0;1.6,0.0;1.6,1.3;0.0,1.3;-1.2,1.0;-1.2,-0.8;0.8,-1.2',
    ),
]


for pose_element in ['x', 'y', 'z', 'yaw']:
    ARGUMENTS.append(
        DeclareLaunchArgument(pose_element, default_value='0.0', description=f'{pose_element} component of robot pose')
    )


def generate_launch_description():
    pkg = get_package_share_directory('task1_package')

    nav_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([PathJoinSubstitution([pkg, 'launch', 'sim_turtlebot_nav.launch.py'])]),
        launch_arguments=[
            ('namespace', LaunchConfiguration('namespace')),
            ('rviz', LaunchConfiguration('rviz')),
            ('world', LaunchConfiguration('world')),
            ('model', LaunchConfiguration('model')),
            ('use_sim_time', LaunchConfiguration('use_sim_time')),
            ('map', LaunchConfiguration('map')),
            ('x', LaunchConfiguration('x')),
            ('y', LaunchConfiguration('y')),
            ('z', LaunchConfiguration('z')),
            ('yaw', LaunchConfiguration('yaw')),
        ],
    )

    mission_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([PathJoinSubstitution([pkg, 'launch', 'autonomous_eval.launch.py'])]),
        launch_arguments=[
            ('use_sim_time', LaunchConfiguration('use_sim_time')),
            ('required_faces', LaunchConfiguration('required_faces')),
            ('required_rings', LaunchConfiguration('required_rings')),
            ('max_mission_seconds', LaunchConfiguration('max_mission_seconds')),
            ('max_search_rounds', LaunchConfiguration('max_search_rounds')),
            ('search_waypoints', LaunchConfiguration('search_waypoints')),
        ],
    )

    ld = LaunchDescription(ARGUMENTS)
    ld.add_action(nav_launch)
    ld.add_action(mission_launch)
    return ld

