cmake_minimum_required(VERSION 3.28)
project(digit-recognition-nn LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# --- Fetch SFML ---
FetchContent_Declare(SFML
    GIT_REPOSITORY https://github.com/SFML/SFML.git
    GIT_TAG 3.0.1
    GIT_SHALLOW ON
    SYSTEM
    EXCLUDE_FROM_ALL
)

# --- Fetch external NeuralNetwork ---
FetchContent_Declare(NeuralNetwork
    GIT_REPOSITORY https://github.com/dredstone1/NeuralNetwork.git
    GIT_TAG main
    GIT_SHALLOW ON
)

FetchContent_MakeAvailable(SFML NeuralNetwork)

# --- Collect source files for the library (all except main) ---
file(GLOB_RECURSE LIB_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

# Optionally, exclude main.cpp from LIB_SOURCES if it exists there
list(FILTER LIB_SOURCES EXCLUDE REGEX ".*main.cpp$")

# --- Create static library from local sources (without main.cpp) ---
add_library(digit_nn STATIC ${LIB_SOURCES})
set_target_properties(digit_nn PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(digit_nn
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(digit_nn
    PUBLIC
        SFML::Graphics
        SFML::Window
        SFML::System
        NeuralNetwork
)

# --- Create executable ---
add_executable(digit-recognition-nn main.cpp)

target_link_libraries(digit-recognition-nn
    PRIVATE
        digit_nn
        SFML::Graphics
        SFML::Window
        SFML::System
        NeuralNetwork
)

message(STATUS "Configured digit-recognition-nn executable")

