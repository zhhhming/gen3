cmake_minimum_required(VERSION 3.8)
project(cartesian_target)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(kdl_parser REQUIRED)
find_package(orocos_kdl REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(xr_interfaces REQUIRED)
find_package(eigen3_cmake_module REQUIRED)
find_package(Eigen3 REQUIRED)

add_executable(cartesian_target_node
  src/cartesian_target_node.cpp
)

# 头文件路径（Eigen3）
target_include_directories(cartesian_target_node PRIVATE
  ${EIGEN3_INCLUDE_DIRS}
)

# 依赖（会自动加 include 和链接）
ament_target_dependencies(cartesian_target_node
  rclcpp
  geometry_msgs 
  sensor_msgs 
  std_msgs
  kdl_parser 
  orocos_kdl 
  tf2 
  tf2_ros
  xr_interfaces
)

# 显式链接 Eigen3（有些环境需要）
target_link_libraries(cartesian_target_node
  Eigen3::Eigen
)

install(TARGETS cartesian_target_node
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
