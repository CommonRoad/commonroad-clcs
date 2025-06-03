# Find omp with apple clang compiler
if (APPLE)
    message(STATUS "Looking for OpenMP with apple clang")
    if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "arm64")
        set(MAC_LIBOMP_PATH /opt/homebrew/opt/libomp)
    else ()
        set(MAC_LIBOMP_PATH /usr/local/opt/libomp)
    endif ()

    list(APPEND CMAKE_PREFIX_PATH ${MAC_LIBOMP_PATH})

    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        foreach(_lang IN ITEMS C CXX)
            set(OpenMP_${_lang}_LIB_NAMES "omp")
            set(OpenMP_${_lang}_FLAGS "-Xclang -fopenmp")
            set(OpenMP_${_lang}_INCLUDE_DIR ${MAC_LIBOMP_PATH}/include)
        endforeach()
    endif()

    find_library(OpenMP_omp_LIBRARY
        NAMES omp
        PATHS ${MAC_LIBOMP_PATH}/lib
    )

    set(CMAKE_DISABLE_PRECOMPILE_HEADERS ON)
endif ()

find_package(OpenMP REQUIRED)
