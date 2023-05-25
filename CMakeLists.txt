cmake_minimum_required(VERSION 3.20)

project(softee LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 23)

find_package(Eigen3 REQUIRED)

find_package(spdlog REQUIRED)
add_compile_definitions(SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_DEBUG)

add_subdirectory(examples)
