#Project
cmake_minimum_required(VERSION 2.4)

if(COMMAND cmake_policy)
      cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)
	  
project(LRTV)

# Flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -msse4.2 -std=c++11")
#if (APPLE)
#set(CMAKE_CXX_FLAGS_RELEASE "-O3 -msse4.2 -stdlib=libstdc++")
#endif (APPLE)

set(CMAKE_BUILD_TYPE Release)

find_package( OpenCV REQUIRED )
# add_library(mgl SHARED IMPORTED)
# set_property(TARGET mgl PROPERTY IMPORTED_LOCATION /usr/local/Cellar/mathgl/2.3.3_1/lib/libmgl.7.4.0.dylib)
# add_library(mgl1 SHARED IMPORTED)
# set_property(TARGET mgl1 PROPERTY IMPORTED_LOCATION /usr/local/Cellar/mathgl/2.3.3_1/lib/libmgl.7.4.0.dylib)
# depthInpaint
file(GLOB LRTV_SRC_FILES "src/*.cpp")
add_executable(depthInpainting ${LRTV_SRC_FILES})
target_link_libraries(depthInpainting ${OpenCV_LIBS})


