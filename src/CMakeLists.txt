rock_library(dem_generation
    SOURCES Dummy.cpp
    HEADERS Dummy.hpp)

rock_executable(dem_generation_bin Main.cpp
    DEPS dem_generation)

