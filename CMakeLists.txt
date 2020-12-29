cmake_minimum_required(VERSION 2.8.3)
project(msf_localization)

set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )	


FIND_PACKAGE( OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()


################################################
## Declare ROS messages, services and actions ##
################################################

## To declare and build messages, services or actions from within this
## package, follow these steps:
## * Let MSG_DEP_SET be the set of packages whose message types you use in
##   your messages/services/actions (e.g. std_msgs, actionlib_msgs, ...).
## * In the file package.xml:
##   * add a build_depend tag for "message_generation"
##   * add a build_depend and a exec_depend tag for each package in MSG_DEP_SET
##   * If MSG_DEP_SET isn't empty the following dependency has been pulled in
##     but can be declared for certainty nonetheless:
##     * add a exec_depend tag for "message_runtime"
## * In this file (CMakeLists.txt):
##   * add "message_generation" and every package in MSG_DEP_SET to
find_package(catkin REQUIRED COMPONENTS
  roscpp
  pcl_conversions
  pcl_msgs
  pcl_ros
  sensor_msgs
  ndt_omp
)

find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED QUIET)

include(cmake/glog.cmake)

# Add GeographicLib.
add_subdirectory(third_party/GeographicLib)

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES msf_localization
#  CATKIN_DEPENDS other_catkin_pkg
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include ${catkin_INCLUDE_DIRS}
  third_party/GeographicLib/include
# include
# ${catkin_INCLUDE_DIRS}
)

include_directories(
            include 
            ${PCL_INCLUDE_DIRS}
            ${OpenCV_INCLUDE_DIRS}
            ${EIGEN3_INCLUDE_DIR}
            )

link_directories(
  include
  ${PCL_LIBRARY_DIRS}
  ${OpenCV_LIBRARY_DIRS}
  )


# 生成共享库  .so  
add_library(${PROJECT_NAME} SHARED
  src/initializer.cpp
)

target_link_libraries(${PROJECT_NAME}
  ${OpenCV_LIBS}
  ${EIGEN3_LIBS}
  ${PCL_LIBRARIES}
  ${GLOG_LIBRARIES}
  libGeographiccc
)


add_executable(simple_localization_node
  app/simple_localization_node.cpp
)
add_dependencies(simple_localization_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(simple_localization_node
  ${catkin_LIBRARIES}
  ${PROJECT_NAME}
)

add_executable(ins_localization_node
  app/gps_imu_localization_node.cpp
)

add_dependencies(ins_localization_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(ins_localization_node
  ${catkin_LIBRARIES}
  ${PROJECT_NAME}
)


add_executable(MapServer_node
  app/MapServer_node.cpp
)
add_dependencies(MapServer_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(MapServer_node
  ${catkin_LIBRARIES}
  ${PROJECT_NAME}
)

add_executable(preprocess_node
  app/preprocess_node.cpp
)
add_dependencies(preprocess_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(preprocess_node
  ${catkin_LIBRARIES}
  ${PROJECT_NAME}
)


