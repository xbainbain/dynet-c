cmake_minimum_required(VERSION 2.8)
project(dynet-c)

set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_STANDARD 11)

set(SOURCES c_api.cc)

include_directories(.)
include_directories(../../)

link_directories(../../build/dynet)

add_library(cdynet SHARED ${SOURCES})
target_link_libraries(cdynet dynet)

add_executable(test test.c)
target_link_libraries(test cdynet)