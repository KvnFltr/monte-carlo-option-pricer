cmake_minimum_required(VERSION 3.20)

project(mcpricer
    VERSION 1.0.0
    DESCRIPTION "Monte Carlo option pricer benchmarked against Black-Scholes"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(MCP_BUILD_TESTS "Build the unit tests (downloads GoogleTest)" ON)
option(MCP_BUILD_APPS "Build the command-line tools" ON)
option(MCP_USE_OPENMP "Parallelise the simulations with OpenMP when available" ON)
option(MCP_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)

# --------------------------------------------------------------------------- library
add_library(mcpricer
    src/types.cpp
    src/black_scholes.cpp
    src/european.cpp
    src/greeks.cpp
    src/asian.cpp
    src/qmc.cpp)
add_library(mcpricer::mcpricer ALIAS mcpricer)

target_include_directories(mcpricer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(mcpricer PUBLIC cxx_std_20)

function(mcp_set_warnings target)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /permissive- /utf-8)
        if(MCP_WARNINGS_AS_ERRORS)
            target_compile_options(${target} PRIVATE /WX)
        endif()
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Wshadow -Wconversion)
        if(MCP_WARNINGS_AS_ERRORS)
            target_compile_options(${target} PRIVATE -Werror)
        endif()
    endif()
endfunction()
mcp_set_warnings(mcpricer)

if(MCP_USE_OPENMP)
    find_package(OpenMP COMPONENTS CXX)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(mcpricer PUBLIC OpenMP::OpenMP_CXX)
        message(STATUS "mcpricer: OpenMP enabled")
    else()
        message(STATUS "mcpricer: OpenMP not found, simulations run on one thread")
    endif()
endif()

# --------------------------------------------------------------------------- tools
if(MCP_BUILD_APPS)
    add_executable(mc_pricer apps/mc_pricer.cpp)
    target_link_libraries(mc_pricer PRIVATE mcpricer)
    mcp_set_warnings(mc_pricer)

    add_executable(run_experiments apps/run_experiments.cpp)
    target_link_libraries(run_experiments PRIVATE mcpricer)
    mcp_set_warnings(run_experiments)
endif()

# --------------------------------------------------------------------------- tests
if(MCP_BUILD_TESTS)
    enable_testing()
    include(FetchContent)
    FetchContent_Declare(googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    add_executable(mcpricer_tests
        tests/test_normal.cpp
        tests/test_rng.cpp
        tests/test_statistics.cpp
        tests/test_black_scholes.cpp
        tests/test_european.cpp
        tests/test_greeks.cpp
        tests/test_asian.cpp
        tests/test_qmc.cpp)
    target_link_libraries(mcpricer_tests PRIVATE mcpricer GTest::gtest_main)
    mcp_set_warnings(mcpricer_tests)

    include(GoogleTest)
    gtest_discover_tests(mcpricer_tests DISCOVERY_TIMEOUT 60)

    if(MCP_BUILD_APPS)
        add_test(NAME cli_smoke COMMAND mc_pricer --paths 200000 --greeks)
    endif()
endif()
