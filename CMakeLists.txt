cmake_minimum_required(VERSION 2.8)

project(rgbseg)

set(CMAKE_CXX_FLAGS "-g -Wall")

# Tell the find_package() macro where to look for OpenCV.
# Only needed if installed files are not in an expected location.
#
# For system-wide install location (default is /usr/local)
if(EXISTS "${CMAKE_INSTALL_PREFIX}/share/OpenCV/OpenCVConfig.cmake")
  set(OpenCV_DIR ${CMAKE_INSTALL_PREFIX}/share/OpenCV)
# For local/custom OpenCV install location (modify/add as needed)
elseif(EXISTS "$ENV{HOME}/opencv/install/share/OpenCV/OpenCVConfig.cmake")
  set(OpenCV_DIR $ENV{HOME}/opencv/install/share/OpenCV)
elseif(EXISTS "$ENV{HOME}/sw/OpenCV3.1/share/OpenCV/OpenCVConfig.cmake")
  set(OpenCV_DIR $ENV{HOME}/sw/OpenCV3.1/share/OpenCV)
endif()

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
