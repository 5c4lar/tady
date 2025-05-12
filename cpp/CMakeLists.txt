include(FetchContent)
# Boost Graph Library
set(BOOST_INCLUDE_LIBRARIES graph)
set(BOOST_ENABLE_CMAKE ON)

find_package(Boost QUIET COMPONENTS ${BOOST_INCLUDE_LIBRARIES})
if(NOT Boost_FOUND)
  message(STATUS "Boost not found, using FetchContent to download it")
  FetchContent_Declare(
    Boost
    GIT_REPOSITORY https://github.com/boostorg/boost.git
    GIT_PROGRESS TRUE
    GIT_TAG boost-1.83.0
  )
  FetchContent_MakeAvailable(Boost)
else()
  message(STATUS "Reusing system Boost")
endif()

include_directories(${Boost_INCLUDE_DIRS})

# Pybind11
find_package(Python3 COMPONENTS Interpreter Development)
find_package(pybind11 QUIET)
if(NOT pybind11_FOUND)
    message(STATUS "pybind11 not found, using FetchContent to download it")
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.11.1  # adjust to the version you need
    )
    FetchContent_MakeAvailable(pybind11)
else()
    message(STATUS "Reusing system pybind11")
endif()


set(LLVM_LINK_LLVM_DYLIB ON)
find_package(LLVM REQUIRED CONFIG)
if(LLVM_FOUND)
  message(STATUS "Using system LLVM ${LLVM_PACKAGE_VERSION}")
  message(STATUS "LLVM_INCLUDE_DIRS: ${LLVM_INCLUDE_DIRS}")
  message(STATUS "LLVM_DIR: ${LLVM_DIR}")

  # Add definitions for shared libraries
  add_definitions(${LLVM_DEFINITIONS})
# include_directories(AFTER ${LLVM_INCLUDE_DIRS})
else()
  message(STATUS "LLVM not found")
endif()

include_directories(include)
# Rename the module to tady_cpp to match Python import conventions
pybind11_add_module(tady_cpp src/tady.cc)
target_link_libraries(tady_cpp PRIVATE LLVM)
target_compile_options(tady_cpp PRIVATE -static-libstdc++)


# Install the module to the correct location
if(SKBUILD_CORE OR SKBUILD)
    # When using scikit-build-core or scikit-build, install to the current directory as
    # CMAKE_INSTALL_PREFIX will be set to the proper Python package directory
    install(TARGETS tady_cpp DESTINATION ".")
    message(STATUS "Installing tady_cpp module to: ${SKBUILD_PROJECT_NAME}")
else()
    # For direct CMake builds, use the Python installation directory
    install(TARGETS tady_cpp DESTINATION ${PYTHON_INSTALL_DIR})
endif()
