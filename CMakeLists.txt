
project(Poulpy)
cmake_minimum_required(VERSION 3.0)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/modules/")
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_SOURCE_DIR}")
set(LIBRARY_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/lib/${CMAKE_BUILD_TYPE}")

set(BUILD_SHARED_LIBS OFF)
find_package(OpenAL REQUIRED)
find_package(ALUT REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(GLUT REQUIRED)
find_package(Qt4 REQUIRED QtCore QtGui QtXml QtOpenGL)
find_package(QGLViewer REQUIRED)
find_package(Log4cpp REQUIRED)
find_package(CUDA REQUIRED 6.5)
include(CUDA_compute_capability)

#-Wshadow -Wstrict-aliasing -Weffc++ -Werror
set(CMAKE_CXX_FLAGS "-W -Wall -Wextra -Wno-unused-parameter -pedantic -std=c++11 -m64")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
SET(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

set(CMAKE_EXE_LINKER_FLAGS "-O2")

set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_NVCC_FLAGS "-arch=compute_${CUDA_COMPUTE_CAPABILITY};-code=sm_${CUDA_COMPUTE_CAPABILITY}")
set(CUDA_NVCC_FLAGS_DEBUG "-G;-g;-Xptxas=-v")
set(CUDA_NVCC_FLAGS_RELEASE "-O3")

#Find source files
file(GLOB_RECURSE header_files src/*.h)
file(GLOB_RECURSE source_files src/*.cpp)
file(GLOB_RECURSE template_files src/*.tpp)
file(GLOB_RECURSE kernel_files src/*.cu)

#Local includes
foreach(header ${header_files})
    get_filename_component(dir ${header} DIRECTORY)
    list(APPEND LOCAL_INCLUDE_DIRS ${dir})
endforeach()
list(REMOVE_DUPLICATES LOCAL_INCLUDE_DIRS)

#Header directories
include_directories(
    ${LOCAL_INCLUDE_DIRS} 
    ${OPENAL_INCLUDE_DIR} 
    ${ALUT_INCLUDE_DIRS} 
    ${OPENGL_INCLUDE_DIR}
    ${GLEW_INCLUDE_DIRS} 
    ${GLUT_INCLUDE_DIR}
    ${QGLVIEWER_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS} 
    ${LOG4CPP_INCLUDE_DIRS} 
)

#Library directories
link_directories(
    ${CUDA_LIBRARY_DIRS}
    ${LOG4CPP_LIBRARY_DIRS}
)

#Precompiled CUDA kernels
set(CUDA_KERNELS "CudaKernels")
cuda_include_directories(${include_directories})
cuda_add_library(${CUDA_KERNELS} ${header_files} ${kernel_files})

#Executables
add_executable(main ${header_files} ${source_files} ${template_files})

#Link executable
target_link_libraries(
    main 
    ${OPENAL_LIBRARY}
    ${ALUT_LIBRARIES}
    ${OPENGL_LIBRARIES}
    ${GLEW_LIBRARIES} 
    ${GLUT_glut_LIBRARY}
    ${QGLVIEWER_LIBRARY}
    Qt4::QtGui 
    Qt4::QtCore 
    Qt4::QtXml 
    Qt4::QtOpenGL
    ${CUDA_LIBRARIES} 
    ${LOG4CPP_LIBRARIES}
    ${CUDA_KERNELS}
)



