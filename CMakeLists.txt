cmake_minimum_required(VERSION 3.10)
project(CNNCPP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

set(ENV{OMP_SCHEDULE} "dynamic")
set(ENV{OMP_PROC_BIND} "true")
set(ENV{OMP_PLACES} "cores")
set(ENV{OMP_MAX_ACTIVE_LEVELS} "3")

if(MSVC)
    set(OPTIMIZATION_FLAGS "/O2 /Oi /GL /Gy /Ot /arch:AVX2 /fp:fast /Qpar /Qpar-report:2")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /LTCG")
else()
    set(CMAKE_CXX_FLAGS_RELEASE "-O3")
    set(OPTIMIZATION_FLAGS "-march=native -mtune=native -mavx2 -mfma -ffast-math -funroll-loops -flto -fomit-frame-pointer -finline-functions -ftree-vectorize")
endif()

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${OPTIMIZATION_FLAGS}")

set(COMMON_SOURCES
    src/cnn.cpp
    src/evaluate.cpp
    src/neural_network.cpp
)

set(HEADERS
    include/cnn.hpp
    include/evaluate.hpp
    include/neural_network.hpp
)

function(set_executable_properties target_name)
    target_include_directories(${target_name} PRIVATE include)

    if(OpenMP_CXX_FOUND)
        target_link_libraries(${target_name} PRIVATE OpenMP::OpenMP_CXX)
    endif()

    set_target_properties(${target_name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endfunction()

add_executable(${PROJECT_NAME} src/cnn_main.cpp ${COMMON_SOURCES} ${HEADERS})
add_executable(evaluate src/evaluate_main.cpp ${COMMON_SOURCES} ${HEADERS})
add_executable(neural_network_train src/neural_network_main.cpp ${COMMON_SOURCES} ${HEADERS})

set_executable_properties(${PROJECT_NAME})
set_executable_properties(evaluate)
set_executable_properties(neural_network_train)

add_custom_command(
    TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/data
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/data ${CMAKE_BINARY_DIR}/data
)

add_custom_target(run_cnn
    COMMAND ${CMAKE_COMMAND} -E env 
        OMP_NUM_THREADS=8
        OMP_SCHEDULE=dynamic 
        OMP_PROC_BIND=true 
        OMP_PLACES=cores 
        $<TARGET_FILE:${PROJECT_NAME}> --train-size 60000 --test-size 10000 --epochs 5 --batch-size 64 ## can modify these hyperparameters
    DEPENDS ${PROJECT_NAME}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running CNN training with optimal settings"
)

add_custom_target(run_evaluate
    COMMAND ${CMAKE_COMMAND} -E env 
        OMP_NUM_THREADS=8
        OMP_SCHEDULE=dynamic 
        OMP_PROC_BIND=true 
        OMP_PLACES=cores 
        $<TARGET_FILE:evaluate> 10000
    DEPENDS evaluate
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running CNN evaluation with optimal settings"
)

add_custom_target(run_neural_network
    COMMAND ${CMAKE_COMMAND} -E env 
        OMP_NUM_THREADS=8
        OMP_SCHEDULE=dynamic 
        OMP_PROC_BIND=true 
        OMP_PLACES=cores 
        $<TARGET_FILE:neural_network_train> --train-size 60000 --test-size 10000 --epochs 5 --batch-size 64 ## can modify these hyperparameters
    DEPENDS neural_network_train
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running Neural Network training with optimal settings"
)