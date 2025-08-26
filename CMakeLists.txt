cmake_minimum_required(VERSION 3.28)
set(CMAKE_CUDA_ARCHITECTURES 86)  # For RTX 3060

project(digit-recognition-nn LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)                    # Add CUDA standard
set(CMAKE_CUDA_STANDARD_REQUIRED ON)           # Enforce it
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_language(CUDA)
include(FetchContent)

FetchContent_Declare(SFML
    GIT_REPOSITORY https://github.com/SFML/SFML.git
    GIT_TAG 3.0.1
    GIT_SHALLOW ON
)

FetchContent_Declare(NeuralNetwork
    GIT_REPOSITORY https://github.com/dredstone1/NeuralNetwork.git
    GIT_TAG main
    GIT_SHALLOW ON
)

FetchContent_MakeAvailable(SFML NeuralNetwork)

file(GLOB_RECURSE LIB_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_CURENT_SOURCE_DIR}/src/*.cu"
)

list(FILTER LIB_SOURCES EXCLUDE REGEX ".*main.cpp$")

add_library(digit_nn STATIC ${LIB_SOURCES})
set_target_properties(digit_nn PROPERTIES POSITION_INDEPENDENT_CODE ON)

set_target_properties(NeuralNetwork PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_include_directories(digit_nn
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

find_package(CUDAToolkit REQUIRED)

target_link_libraries(digit_nn
    PUBLIC
        sfml-graphics
        sfml-window
        sfml-system
        NeuralNetwork
        CUDA::cudart
)

add_executable(digit-nn src/digit_recognition.cpp)
add_executable(abc-digit-nn src/abc_digit_recognition.cpp)

target_link_libraries(digit-nn
    PRIVATE
        digit_nn
        sfml-graphics
        sfml-window
        sfml-system
        CUDA::cudart
)

target_link_libraries(abc-digit-nn
    PRIVATE
        digit_nn
        sfml-graphics
        sfml-window
        sfml-system
        CUDA::cudart
)

message(STATUS "Configured image-recognition-nn executable")
