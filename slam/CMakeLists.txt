cmake_minimum_required(VERSION 3.0)
project(ch6)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-std=c++11 -O3")

find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})


find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

include_directories("/usr/local/Cellar/eigen/3.4.0_1/include/eigen3")

add_executable(newton newton.cpp)
add_executable(basiccv basiccv.cpp)
add_executable(undistort undistortImage.cpp)
add_executable(stereo stereoVision.cpp)
add_executable(joinMap joinMap.cpp)


target_link_libraries(basiccv ${OpenCV_LIBS})
target_link_libraries(newton ${OpenCV_LIBS} ${CERES_LIBRARIES})
target_link_libraries(undistort ${OpenCV_LIBS})
target_link_libraries(stereo ${OpenCV_LIBS} ${Pangolin_LIBRARIES})
target_link_libraries(joinMap ${OpenCV_LIBS} ${Pangolin_LIBRARIES} Sophus::Sophus)

