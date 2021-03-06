cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(dynet-c)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 11)

set(SOURCES c_api.cc)
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/examples)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

if(NOT DEFINED DYNET_INCLUDE_DIR)
    message(FATAL_ERROR "-- DYNET_INCLUDE_DIR not specified")
endif()

include_directories(${DYNET_INCLUDE_DIR})
link_directories(${DYNET_INCLUDE_DIR}/build/dynet)

add_library(dynetc SHARED ${SOURCES})
target_link_libraries(dynetc dynet)

add_executable(test test.c)
target_link_libraries(test dynetc)