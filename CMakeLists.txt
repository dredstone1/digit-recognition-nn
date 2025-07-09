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

# --- Collect source files for this project ---
file(GLOB_RECURSE PROJECT_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

# --- Create static library from local sources ---
add_library(digit_nn STATIC ${PROJECT_SOURCES})
set_target_properties(digit_nn PROPERTIES POSITION_INDEPENDENT_CODE ON)

# --- Include directories ---
target_include_directories(digit_nn
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# --- Link external libraries ---
target_link_libraries(digit_nn
    PUBLIC
        SFML::Graphics
        SFML::Window
        SFML::System
        NeuralNetwork
)

message(STATUS "Configured digit-recognition-nn")
