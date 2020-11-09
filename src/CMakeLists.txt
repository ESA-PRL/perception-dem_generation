find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

rock_library(dem_generation
    SOURCES dem_generation.cpp
    HEADERS dem_generation.hpp
    DEPS_PKGCONFIG base-lib ${OPENCV_PACKAGE})

rock_executable(dem_generation_bin Main.cpp
    DEPS dem_generation)

if(NOT "${PCL_LIBRARIES}" STREQUAL "")
  list(REMOVE_ITEM PCL_LIBRARIES "vtkproj4")
endif()

target_link_libraries(dem_generation
  ${PCL_LIBRARIES}
  opencv_core)
