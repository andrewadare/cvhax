cmake_minimum_required(VERSION 2.8)

project(rgbseg)

set(CMAKE_CXX_FLAGS "-g -Wall")

# If the find_package() macro fails to locate your OpenCV installation, or if
# you wish to use a "custom" OpenCV build, here are two solutions:
#   1. Set OpenCV_DIR as an environment variable.
#      export OpenCV_DIR=/path/to/OpenCVConfig.cmake.
#   2. Set OpenCV_DIR as a CMake variable here, e.g.
#      if(EXISTS "${CMAKE_INSTALL_PREFIX}/share/OpenCV/OpenCVConfig.cmake")
#        set(OpenCV_DIR ${CMAKE_INSTALL_PREFIX}/share/OpenCV)
#      endif()
# The OpenCVConfig.cmake file contains some useful build info.

find_package(OpenCV REQUIRED)
message("\nOpenCV_DIR: " ${OpenCV_DIR})

include_directories(${OpenCV_INCLUDE_DIRS})

set(SRC ${PROJECT_SOURCE_DIR}/src)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

# Get list of .cpp files. CMake must be re-run whenever new files are added.
file(GLOB MY_CPPS RELATIVE ${SRC} ${SRC}/*.cpp)

# Make an executable from each .cpp file
message("\nExecutables to be built:")
foreach(CPPFILE ${MY_CPPS})
  # Extract foo from foo.cpp as exename
  string(REPLACE ".cpp" "" exename ${CPPFILE})
  # Generate foo executable
  add_executable(${exename} ${SRC}/${CPPFILE})
  target_link_libraries(${exename} ${OpenCV_LIBS})
  # Print executable list
  message("  ${exename}")
endforeach()

message("\nExecutables will be installed in ${EXECUTABLE_OUTPUT_PATH}.\n")
