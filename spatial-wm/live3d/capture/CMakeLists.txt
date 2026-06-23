cmake_minimum_required(VERSION 3.16)
project(desktopcapture LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(desktopcapture SHARED capture.cpp)

find_package(X11 REQUIRED)
target_link_libraries(desktopcapture PRIVATE
    ${X11_LIBRARIES} ${X11_Xcomposite_LIB} GL)
target_include_directories(desktopcapture PRIVATE ${X11_INCLUDE_DIR})
