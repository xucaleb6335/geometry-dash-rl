# GD Agent Control mod. Needs the Geode SDK (GEODE_SDK env var), Clang/MSVC
# and the Windows SDK (see ../AGENTS.md). Build with:  geode build   (from this
# folder), or: python -m gdctl build control-mod
cmake_minimum_required(VERSION 3.29)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)

project(GDAgentControl VERSION 0.1.0)

add_library(${PROJECT_NAME} SHARED
    src/main.cpp
    src/Commands.cpp
    src/LineServer.cpp
)

if (NOT DEFINED ENV{GEODE_SDK})
    message(FATAL_ERROR "Unable to find Geode SDK! Set the GEODE_SDK environment variable (geode sdk install does this).")
else()
    message(STATUS "Found Geode: $ENV{GEODE_SDK}")
endif()

add_subdirectory($ENV{GEODE_SDK} ${CMAKE_CURRENT_BINARY_DIR}/geode)

setup_geode_mod(${PROJECT_NAME})

# Same trick as mod/: Geode's forced PCH pulls in windows.h (and winsock.h)
# before winsock2.h. LineServer is Geode-free, so build it without the PCH.
set_source_files_properties(src/LineServer.cpp PROPERTIES SKIP_PRECOMPILE_HEADERS ON)

if (WIN32)
    target_link_libraries(${PROJECT_NAME} ws2_32)
endif()
