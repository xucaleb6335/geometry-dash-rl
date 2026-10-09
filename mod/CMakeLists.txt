# Geode mod build. Needs the Geode SDK (GEODE_SDK env var), Clang + Ninja and the
# Windows SDK; see the README. Build with:  geode build   (from this folder)
cmake_minimum_required(VERSION 3.29)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_OSX_ARCHITECTURES "arm64;x86_64")

project(GDRLBridge VERSION 0.1.0)

add_library(${PROJECT_NAME} SHARED
    src/main.cpp
    src/Controller.cpp
    src/StateReader.cpp
    src/Bridge.cpp
)

if (NOT DEFINED ENV{GEODE_SDK})
    message(FATAL_ERROR "Unable to find Geode SDK! Set the GEODE_SDK environment variable (geode sdk install does this).")
else()
    message(STATUS "Found Geode: $ENV{GEODE_SDK}")
endif()

add_subdirectory($ENV{GEODE_SDK} ${CMAKE_CURRENT_BINARY_DIR}/geode)

setup_geode_mod(${PROJECT_NAME})

# Geode force-includes Geode.hpp as a precompiled header, which pulls in
# windows.h (and the old winsock.h) before Bridge.cpp can include winsock2.h.
# Bridge is Geode-free, so build it without the PCH.
set_source_files_properties(src/Bridge.cpp PROPERTIES SKIP_PRECOMPILE_HEADERS ON)

if (WIN32)
    target_link_libraries(${PROJECT_NAME} ws2_32)
endif()
