cmake_minimum_required(VERSION 3.15)
project(DualMS)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
  set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
endif()

find_package(glfw3 CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(OpenGL REQUIRED)
find_package(CUDAToolkit REQUIRED)

enable_language(CUDA)

add_executable(main 
    main.cpp
    gCVT/gCVTHost.cu
)

set_property(TARGET main PROPERTY CUDA_STANDARD 20)
set_property(TARGET main PROPERTY CUDA_STANDARD_REQUIRED ON)

target_link_libraries(main 
    glfw 
    OpenGL::GL
    glm::glm
    CUDA::cudart
    CUDA::cuda_driver
)

target_include_directories(main PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/gCVT
)

target_compile_definitions(main PRIVATE _CRT_SECURE_NO_WARNINGS)