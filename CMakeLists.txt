cmake_minimum_required(VERSION 2.8)

project(ocv)
# find_package(OpenCV REQUIRED)

macro(LIST_REPLACE LIST INDEX NEWVALUE)
    list(INSERT ${LIST} ${INDEX} ${NEWVALUE})
    MATH(EXPR __INDEX "${INDEX} + 1")
    list (REMOVE_AT ${LIST} ${__INDEX})
endmacro(LIST_REPLACE)

set(SRC ${PROJECT_SOURCE_DIR}/src)
set (CVDIR $ENV{HOME}/opencv/install)
include_directories(${CVDIR}/include)
link_directories(${CVDIR}/lib)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

# Get list of .cpp source files. 
# CMake must be run if any new files are added.
file(GLOB MY_CPPS RELATIVE ${SRC} ${SRC}/*.cpp)


# Print list of include and library search directories
message("\nInclude directories:")
get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
foreach(dir ${dirs})
  message(" ${dir}")
endforeach()
message("\nLibrary directories:")
get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY LINK_DIRECTORIES)
foreach(dir ${dirs})
  message(" ${dir}")
endforeach()

# Get list of shared libraries, and modify libopencv_foo.so to opencv_foo.
# This a hack to get the equivalent of OpenCV_LIBS.
file(GLOB CV_LIBS RELATIVE ${CVDIR}/lib ${CVDIR}/lib/*.so)
foreach(lib ${CV_LIBS})
  list(FIND CV_LIBS ${lib} idx)
  # message ("${idx} ${lib}")
  string(REPLACE ".so" "" tmp1 ${lib})
  string(REPLACE "libopencv" "opencv" tmp2 ${tmp1})
  LIST_REPLACE(CV_LIBS ${idx} ${tmp2})
endforeach()

# foreach(lib ${CV_LIBS})
#   message (${lib})
# endforeach()
# message("\n")

# message("\nOpenCV_LIBS:")
# message(${OpenCV_LIBS})
# message("\n")

# foreach(lib ${OpenCV_LIBS})
#   message(" ${lib}")
# endforeach()
# message("\n")

# Make an executable from each .cpp file
message("\nExecutables to be built:")
foreach(CPPFILE ${MY_CPPS})
  # Extract foo from foo.cpp as exename
  string(REPLACE ".cpp" "" exename ${CPPFILE})
  # Generate foo executable
  add_executable(${exename} ${SRC}/${CPPFILE})
  target_link_libraries(${exename} ${CV_LIBS})
  # Print executable list
  message(${exename})
endforeach(CPPFILE ${MY_CPPS})

message("\nExecutables will be installed in ${EXECUTABLE_OUTPUT_PATH}.\n")
