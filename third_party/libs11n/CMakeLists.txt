project(s11n)
cmake_minimum_required(VERSION 3.1)

option(BUILD_SHARED_LIBS "Build s11n as a shared library" ON)

set(CMAKE_CXX_STANDARD 17)

set(S11_SRC_ROOT "./src")

set(S11_SRC 	

${S11_SRC_ROOT}/argv_parser.cpp
${S11_SRC_ROOT}/compact.flex.cpp
${S11_SRC_ROOT}/compact_serializer.cpp
${S11_SRC_ROOT}/data_node_io.cpp
${S11_SRC_ROOT}/exception.cpp
${S11_SRC_ROOT}/funtxt.flex.cpp
${S11_SRC_ROOT}/funtxt_serializer.cpp
${S11_SRC_ROOT}/funxml.flex.cpp
${S11_SRC_ROOT}/funxml_serializer.cpp
${S11_SRC_ROOT}/parens.flex.cpp
${S11_SRC_ROOT}/parens_serializer.cpp
${S11_SRC_ROOT}/path_finder.cpp
${S11_SRC_ROOT}/plugin.cpp
${S11_SRC_ROOT}/s11n.cpp
${S11_SRC_ROOT}/s11nlite.cpp
${S11_SRC_ROOT}/s11n_node.cpp
${S11_SRC_ROOT}/simplexml.flex.cpp
${S11_SRC_ROOT}/simplexml_serializer.cpp
${S11_SRC_ROOT}/strtool.cpp
${S11_SRC_ROOT}/wesnoth.flex.cpp
${S11_SRC_ROOT}/wesnoth_serializer.cpp
)

file(GLOB_RECURSE S11_HEADERS ${PROJECT_SOURCE_DIR}/include/*.hpp)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# IMPORTANT NOTE:
#
# Building s11n as a static library WON'T WORK and will SILENTLY BREAK serialization
# When linking against a static library, the linker will only add symbols which are
# actually used in the resulting executable/shared library.
# However, the plugin system used by s11n needs all s11n symbols to be included,
# even when some plugin symbols appear to be unused.
#
# As a result, it is only possible to use s11n as a shared library or an object library.
# The latter option will result in s11n's object files being included directly
# (without creating a static archive first).

add_library(s11n OBJECT ${S11_SRC})

add_library(s11n::s11n ALIAS s11n)

set_property(TARGET s11n PROPERTY POSITION_INDEPENDENT_CODE ON)

target_compile_options(s11n PUBLIC $<$<CXX_COMPILER_ID:GNU,Clang>:-Wno-deprecated-declarations>)

target_include_directories(s11n
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
  )

install(
    DIRECTORY ${PROJECT_SOURCE_DIR}/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )

install(TARGETS s11n
    EXPORT ${PROJECT_NAME}_Targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}Config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
  INSTALL_DESTINATION
    ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
  )

# Export target configuration (for installation)
install(EXPORT ${PROJECT_NAME}_Targets
    FILE ${PROJECT_NAME}Targets.cmake
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
  )

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
  )