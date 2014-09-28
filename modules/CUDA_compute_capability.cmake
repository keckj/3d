#######################################################
# Check for GPUs present and their compute capability #
#######################################################

if(CUDA_FOUND)
    
    message("-- Checking available GPUs and acrhitecture...")
    
    try_run(RUN_RESULT_VAR COMPILE_RESULT_VAR
        ${CMAKE_BINARY_DIR} 
        "${CMAKE_CURRENT_SOURCE_DIR}/modules/CUDA_compute_capability.c"
        CMAKE_FLAGS 
        -DINCLUDE_DIRECTORIES:STRING=${CUDA_TOOLKIT_INCLUDE}
        -DLINK_LIBRARIES:STRING=${CUDA_CUDART_LIBRARY}
        COMPILE_OUTPUT_VARIABLE COMPILE_OUTPUT_VAR
        RUN_OUTPUT_VARIABLE RUN_OUTPUT_VAR)

    # COMPILE_RESULT_VAR is TRUE when compile succeeds
    # RUN_RESULT_VAR is zero when a GPU is found

    if(COMPILE_RESULT_VAR AND NOT RUN_RESULT_VAR)
        message("--     CUDA capable GPU found !")
        message("--     Using Compute Unified Device Architecture ${RUN_OUTPUT_VAR}!")
        set(CUDA_HAVE_GPU TRUE CACHE BOOL "Whether CUDA-capable GPU is present")
        set(CUDA_COMPUTE_CAPABILITY ${RUN_OUTPUT_VAR} CACHE STRING "Compute capability of CUDA-capable GPU present")
        mark_as_advanced(CUDA_COMPUTE_CAPABILITY)
    else()
        set(CUDA_HAVE_GPU FALSE CACHE BOOL "Whether CUDA-capable GPU is present")
        message(FATAL_ERROR "--     No CUDA capable GPU found !")
    endif()
endif(CUDA_FOUND)



