cmake_minimum_required(VERSION 3.12)

project(COVID19 LANGUAGES CXX)

set(EXTERN_DIR    "${CMAKE_CURRENT_SOURCE_DIR}/extern")
set(SRC_DIR       "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(CUSTOM_CXXFLAGS -Wall -Wextra -march=native)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)  # Use -std=c++14, not -std=gnu++14.
add_subdirectory(${EXTERN_DIR}/pybind11 epidemics_pybind11)

find_package(Boost)

add_library(
    libepidemics_core
    STATIC
    # For now, bindings are responsible for instantiating template
    # functions `Solver::solve`, so there's not that many files here.
    ${SRC_DIR}/data.cpp
    ${SRC_DIR}/utils/assert.cpp
    ${SRC_DIR}/utils/signal.cpp
)
set_property(TARGET libepidemics_core PROPERTY POSITION_INDEPENDENT_CODE ON)
target_compile_options(libepidemics_core PUBLIC ${CUSTOM_CXXFLAGS})
target_include_directories(libepidemics_core PUBLIC ${SRC_DIR})
# Backwards-compatible alias to Boost::headers.
target_link_libraries(libepidemics_core PUBLIC Boost::boost)

add_library(libepidemics::core ALIAS libepidemics_core)

pybind11_add_module(
    libepidemics
    ${SRC_DIR}/bindings/bindings.cpp
    ${SRC_DIR}/bindings/cantons.seiin_interventions.cpp
)
target_link_libraries(libepidemics PRIVATE libepidemics::core)

# Make Release mode the default.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are: Debug Release
      RelWithDebInfo MinSizeRel."
      FORCE)
endif(NOT CMAKE_BUILD_TYPE)
