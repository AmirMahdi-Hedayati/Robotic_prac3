from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():

    pkg_share = FindPackageShare('hw_robot_description')

    map_yaml = PathJoinSubstitution([
        pkg_share,
        'maps',
        'depot.yaml'
    ])

    map_server_params = PathJoinSubstitution([
        pkg_share,
        'config',
        'map_server.yaml'
    ])

    rviz_config = PathJoinSubstitution([
        pkg_share,
        'rviz',
        'config.rviz'
    ])

    return LaunchDescription([

        Node(
            package='nav2_map_server',
            executable='map_server',
            name='map_server',
            parameters=[
                map_server_params,
                {'yaml_filename': map_yaml},
                {'use_sim_time': True}
            ],
            output='screen'
        ),

                Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_localization',
            output='screen',
            parameters=[{
                'use_sim_time': True,
                'autostart': True,
                'node_names': ['map_server']
            }]
        ),

        Node(
            package='rviz2',
            executable='rviz2',
            parameters=[{'use_sim_time': True}],
            arguments=['-d', rviz_config],
            output='screen'
        ),
    ])