cmake_minimum_required(VERSION 3.16)
project(stereo_mirror LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Reuse the GPU window-capture compositor from the glasses-demo (single source of
# truth — no copy). No Viture SDK needed: this program is device/IMU-free.
get_filename_component(CAPTURE_DIR "${CMAKE_SOURCE_DIR}/../glasses-demo" ABSOLUTE)

# --- raylib (fetched from GitHub) --------------------------------------------
include(FetchContent)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_GAMES    OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_WAYLAND OFF CACHE BOOL "" FORCE)   # force X11 GLFW backend
set(GLFW_BUILD_X11     ON  CACHE BOOL "" FORCE)
FetchContent_Declare(raylib
    GIT_REPOSITORY https://github.com/raysan5/raylib.git
    GIT_TAG        5.5
    GIT_SHALLOW    TRUE)
FetchContent_MakeAvailable(raylib)

add_executable(stereo-mirror
    main_stereo.cpp
    "${CAPTURE_DIR}/window_capture.cpp")
target_include_directories(stereo-mirror PRIVATE "${CAPTURE_DIR}")
target_link_libraries(stereo-mirror PRIVATE raylib)

# X11 + GLX + Composite for GPU window capture (texture_from_pixmap).
find_package(X11 REQUIRED)
target_link_libraries(stereo-mirror PRIVATE
    ${X11_LIBRARIES} ${X11_Xcomposite_LIB} ${X11_Xdamage_LIB}
    ${X11_Xfixes_LIB} ${X11_Xrandr_LIB} GL)
target_include_directories(stereo-mirror PRIVATE ${X11_INCLUDE_DIR})
