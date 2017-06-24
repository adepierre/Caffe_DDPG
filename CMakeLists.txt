cmake_minimum_required(VERSION 3.5)

project(Caffe_DDPG)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin)

#Add Caffe
find_package(Caffe)

#Add source files
file(GLOB src_files "src/*.cpp")
file(GLOB header_files "include/*.h")

include_directories("include")

add_executable(Caffe_DDPG ${src_files} ${header_files})

target_link_libraries(Caffe_DDPG ${Caffe_LIBRARIES})