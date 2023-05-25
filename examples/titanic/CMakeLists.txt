
add_executable(titanic main.cpp)

target_include_directories(titanic
    PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${Eigen3_INCLUDE_DIRS}
    ${spdlog_INCLUDE_DIRS}
)

target_link_libraries(titanic
    PUBLIC
    spdlog::spdlog
)
