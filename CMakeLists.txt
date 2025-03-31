cmake_minimum_required(VERSION 3.20)
project(nanograd LANGUAGES CXX CUDA)

# Отключить LTO
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)

# Настройка стандартов
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(PYBIND11_FINDPYTHON ON)

# Загрузка pybind11
include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.11.1
)
FetchContent_MakeAvailable(pybind11)

# Настройка CUDA
find_package(CUDAToolkit REQUIRED)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)  # Для GPU с архитектурой Volta (V100) и новее
endif()

# Библиотека core_lib
add_library(core_lib STATIC
    src/core/tensor.cpp  # <-- Исправленная опечатка
    src/core/ops.cpp
    src/cuda/ops.cu
)

set_target_properties(core_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(core_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

target_link_libraries(core_lib PUBLIC
    CUDA::cudart
)

# Модуль Python
pybind11_add_module(nanograd src/bindings/btensor.cpp)

target_include_directories(nanograd PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${pybind11_INCLUDE_DIRS}
)

target_link_libraries(nanograd PRIVATE
    core_lib
    pybind11::module
)

install(TARGETS nanograd DESTINATION .)