from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    arguments = [
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('required_faces', default_value='3'),
        DeclareLaunchArgument('required_rings', default_value='2'),
        DeclareLaunchArgument('max_mission_seconds', default_value='480.0'),
        DeclareLaunchArgument('max_search_rounds', default_value='2'),
        DeclareLaunchArgument(
            'search_waypoints',
            default_value='0.0,0.0;1.6,0.0;1.6,1.3;0.0,1.3;-1.2,1.0;-1.2,-0.8;0.8,-1.2'
        ),
    ]

    detect_faces = Node(
        package='task1_package',
        executable='detect_people.py',
        name='detect_faces',
        output='screen',
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
            {'detector_backend': 'haar'},
            {'face_vote_threshold': 6},
            {'merge_distance': 0.6},
        ],
    )

    detect_rings = Node(
        package='task1_package',
        executable='extract_color_from_pointcloud.py',
        name='detect_rings',
        output='screen',
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
            {'ring_vote_threshold': 4},
            {'merge_distance': 0.7},
        ],
    )

    commander = Node(
        package='task1_package',
        executable='robot_commander.py',
        name='robot_commander',
        output='screen',
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
            {'required_faces': LaunchConfiguration('required_faces')},
            {'required_rings': LaunchConfiguration('required_rings')},
            {'max_mission_seconds': LaunchConfiguration('max_mission_seconds')},
            {'max_search_rounds': LaunchConfiguration('max_search_rounds')},
            {'search_waypoints': LaunchConfiguration('search_waypoints')},
        ],
    )

    speech_player = Node(
        package='task1_package',
        executable='speech_player.py',
        name='speech_player',
        output='screen',
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
        ],
    )

    ld = LaunchDescription(arguments)
    ld.add_action(detect_faces)
    ld.add_action(detect_rings)
    ld.add_action(commander)
    ld.add_action(speech_player)
    return ld

