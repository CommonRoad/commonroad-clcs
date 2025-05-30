# source files
set(CR_CLCS_SRC_FILES
        src/segment.cc
        src/curvilinear_coordinate_system.cc
        src/projection_domain.cc
        src/path_segments.cc
        src/util.cc
        src/util_sweep_line.cc
        src/util_projection_domain.cc
        src/serialize/serialize_curvil_cs.cc
        src/serialize/public/serialize_public.cc
)

# header files
set(CR_CLCS_HDR_FILES
        include/geometry/application_settings.h
        include/geometry/clcs_exceptions.h
        include/geometry/clcs_types.h
        include/geometry/clcs_logger.h
        include/geometry/segment.h
        include/geometry/curvilinear_coordinate_system.h
        include/geometry/projection_domain.h
        include/geometry/path_segments.h
        include/geometry/util.h
        include/geometry/util_sweep_line.h
        include/geometry/util_projection_domain.h
        include/geometry/serialize/basic_types.h
        include/geometry/serialize/curvilinear_coordinate_system_export.h
        include/geometry/serialize/icurvilinear_coordinate_system_export.h
        include/geometry/serialize/serialize.h
        include/geometry/serialize/serialize_reg_impl.h
        include/geometry/serialize/vector2d_export.h
        include/geometry/serialize/vector2d_export_streams.h
        include/geometry/serialize/export_structs/curvilinear_coordinate_system_export_struct.h
        include/geometry/serialize/public/serialize_public.h
)

# add source files
add_library(crccosy ${CR_CLCS_SRC_FILES})

# add header files
target_sources(crccosy PRIVATE ${CR_CLCS_HDR_FILES})

if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.23.0)
    target_sources(crccosy
            INTERFACE
            FILE_SET crccosy_headers
            TYPE HEADERS
            BASE_DIRS include
            FILES ${CR_CLCS_HDR_FILES})
endif ()

# set compile features
target_compile_features(crccosy PUBLIC cxx_lambdas cxx_auto_type)

target_include_directories(crccosy
        PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/cpp/include>
)

# link private: OpenMP
target_link_libraries(crccosy
        PRIVATE
        OpenMP::OpenMP_CXX
)

# link public: Eigen, Boost, spdlog
target_link_libraries(crccosy
        PUBLIC
        Eigen3::Eigen
        Boost::headers
        Boost::geometry
        spdlog::spdlog
)

# link libs11n
if (CR_CLCS_BUILD_S11N)
    message(STATUS "<CommonRoadCLCS> Serialization and pickling support have been ENABLED for crccosy")
    target_link_libraries(crccosy PUBLIC s11n::s11n)
    target_compile_definitions(crccosy PRIVATE -DENABLE_SERIALIZER=1)
else ()
    message(STATUS "<CommonRoadCLCS> Serialization and pickling support have been DISABLED for crccosy")
    target_compile_definitions(crccosy PRIVATE -DENABLE_SERIALIZER=0)
endif ()

add_library(CommonRoadCLCS::crccosy ALIAS crccosy)

if (${PROJECT_NAME}_IS_TOP_LEVEL)
    foreach (target IN ITEMS crccosy)
        # this adds many warnings to the build. They usually help to find some bugs
        # TODO: Check whether each warning flag is actually supported by the compiler before adding it
        if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
            target_compile_options(${target} PRIVATE -Wall -Wextra -Wconversion
                    -pedantic -Wfatal-errors -Wno-unused-parameter -Wno-unused-function -Wno-unused-but-set-variable)
        endif ()
        if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
            target_compile_options(${target} PRIVATE /W3)
        endif ()
    endforeach ()
endif ()
