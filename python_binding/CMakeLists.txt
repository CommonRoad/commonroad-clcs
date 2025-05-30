set(python_module_name pycrccosy)
pybind11_add_module(${python_module_name} MODULE
        py_bind.cc
        module_geometry.cc
)
target_link_libraries(${python_module_name} PRIVATE crccosy)

# link libs11n
if (CR_CLCS_BUILD_S11N)
    message(STATUS "<CommonRoadCLCS> Serialization and pickling support have been ENABLED for pycrccosy")
    target_link_libraries(${python_module_name} PUBLIC s11n::s11n)
    target_compile_definitions(${python_module_name} PRIVATE -DENABLE_SERIALIZER=1)
else ()
    message(STATUS "<CommonRoadCLCS> Serialization and pickling support have been DISABLED for pycrccosy")
    target_compile_definitions(${python_module_name} PRIVATE -DENABLE_SERIALIZER=0)
endif ()

install(TARGETS ${python_module_name}
        LIBRARY DESTINATION commonroad_clcs
        COMPONENT ${python_module_name}
)

add_custom_target(install-python-modules-${python_module_name}
        ${CMAKE_COMMAND}
        -DCMAKE_INSTALL_COMPONENT=${python_module_name}
        -P "${PROJECT_BINARY_DIR}/cmake_install.cmake"
        DEPENDS ${python_module_name}
)
# include(utils/CheckRuntimeDependencies)
# check_python_runtime_dependencies(${python_module_name})

if (SKBUILD)
    # Sanity check: Ensure we are building all dependencies as static libraries
    include(utils/EnsureStatic)
    if (SKBUILD_SELF_CONTAINED)
        # If we are building all dependencies ourselves, then they should
        # all be static.
        ensure_all_static(crccosy)
    else ()
        # If we are partially using system libraries, then only the crccosy library
        # itself has to be static.
        ensure_static(crccosy)
    endif ()
endif ()
