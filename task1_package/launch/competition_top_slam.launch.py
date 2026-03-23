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
    DeclareLaunchArgument('sync', default_value='true', choices=['true', 'false'], description='Use synchronous SLAM'),
    DeclareLaunchArgument('autostart', default_value='true', choices=['true', 'false'], description='Autostart SLAM toolbox'),
    DeclareLaunchArgument('use_lifecycle_manager', default_value='false', choices=['true', 'false'], description='Use lifecycle manager in SLAM'),
    DeclareLaunchArgument(
        'params',
        default_value=PathJoinSubstitution([get_package_share_directory('task1_package'), 'config', 'slam.yaml']),
        description='SLAM parameters file',
    ),
]


for pose_element in ['x', 'y', 'z', 'yaw']:
    ARGUMENTS.append(
        DeclareLaunchArgument(pose_element, default_value='0.0', description=f'{pose_element} component of robot pose')
    )


def generate_launch_description():
    pkg = get_package_share_directory('task1_package')

    sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([PathJoinSubstitution([pkg, 'launch', 'sim.launch.py'])]),
        launch_arguments=[
            ('world', LaunchConfiguration('world')),
            ('use_sim_time', LaunchConfiguration('use_sim_time')),
        ],
    )

    spawn_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([PathJoinSubstitution([pkg, 'launch', 'turtlebot4_spawn.launch.py'])]),
        launch_arguments=[
            ('namespace', LaunchConfiguration('namespace')),
            ('rviz', LaunchConfiguration('rviz')),
            ('model', LaunchConfiguration('model')),
            ('use_sim_time', LaunchConfiguration('use_sim_time')),
            ('x', LaunchConfiguration('x')),
            ('y', LaunchConfiguration('y')),
            ('z', LaunchConfiguration('z')),
            ('yaw', LaunchConfiguration('yaw')),
        ],
    )

    slam_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([PathJoinSubstitution([pkg, 'launch', 'slam.launch.py'])]),
        launch_arguments=[
            ('namespace', LaunchConfiguration('namespace')),
            ('use_sim_time', LaunchConfiguration('use_sim_time')),
            ('sync', LaunchConfiguration('sync')),
            ('autostart', LaunchConfiguration('autostart')),
            ('use_lifecycle_manager', LaunchConfiguration('use_lifecycle_manager')),
            ('params', LaunchConfiguration('params')),
        ],
    )

    ld = LaunchDescription(ARGUMENTS)
    ld.add_action(sim_launch)
    ld.add_action(spawn_launch)
    ld.add_action(slam_launch)
    return ld

