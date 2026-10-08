# Geode-free build of the bridge, for testing the protocol without the game.
# cmake -S mod/tools -B build/tools && cmake --build build/tools
cmake_minimum_required(VERSION 3.21)
project(GDRLTools CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(standalone_server
    standalone_server.cpp
    ../src/Bridge.cpp
)
target_include_directories(standalone_server PRIVATE ../src)

if (WIN32)
    target_link_libraries(standalone_server PRIVATE ws2_32)
endif()
