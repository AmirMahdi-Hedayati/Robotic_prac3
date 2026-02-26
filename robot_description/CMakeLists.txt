cmake_minimum_required(VERSION 3.8)
project(robot_description)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(tf2 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(std_msgs REQUIRED)

add_executable(frame_id_converter_node scripts/frame_id_converter.cpp)
ament_target_dependencies(frame_id_converter_node rclcpp sensor_msgs)

add_executable(motor_command_node scripts/motor_command.cpp)
ament_target_dependencies(motor_command_node rclcpp std_msgs)

add_executable(ekf_diff_imu_node scripts/ekf_diff_imu.cpp)
target_include_directories(ekf_diff_imu_node PRIVATE ${EIGEN3_INCLUDE_DIRS})
ament_target_dependencies(ekf_diff_imu_node
  rclcpp
  nav_msgs
  sensor_msgs
  std_srvs
  tf2
  tf2_ros
  geometry_msgs
)

install(
  DIRECTORY src launch rviz config world scripts
  DESTINATION share/${PROJECT_NAME}
)

install(
    TARGETS 
    frame_id_converter_node
    ekf_diff_imu_node
    motor_command_node
    DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
