from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler, ExecuteProcess
from launch.conditions import LaunchConfigurationNotEquals
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node

ARGUMENTS = [
    DeclareLaunchArgument('namespace', default_value='', description='Robot namespace'),
]

def generate_launch_description():
    this_package = get_package_share_directory('dis_tutorial3')

    namespace = LaunchConfiguration('namespace')

    control_params_file = PathJoinSubstitution(
        [this_package, 'config', 'all_controls_jtc.yaml'])

    diffdrive_controller_node = Node(
        package='controller_manager',
        executable='spawner',
        namespace=namespace,
        parameters=[control_params_file],
        arguments=['diffdrive_controller', '-c', 'controller_manager'],
        output='screen',
    )

    load_diffdrive_controller = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active', 'diffdrive_controller'],
        output='screen'
    )

    arm_controller_node = Node(
        package='controller_manager',
        executable='spawner',
        namespace=namespace,
        parameters=[control_params_file],
        arguments=['arm_controller', '-c', 'controller_manager'],
        output='screen',
    )

    load_arm_controller = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active', 'arm_controller'],
        output='screen'
    )

    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster', '-c', 'controller_manager'],
        output='screen',
    )

    diffdrive_controller_callback = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=joint_state_broadcaster_spawner,
            on_exit=[diffdrive_controller_node],
        )
    )

    arm_controller_callback = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=diffdrive_controller_node,
            on_exit=[arm_controller_node]
        )
    )

    ld = LaunchDescription(ARGUMENTS)

    ld.add_action(joint_state_broadcaster_spawner)
    ld.add_action(diffdrive_controller_callback)
    ld.add_action(arm_controller_callback)

    return ld
