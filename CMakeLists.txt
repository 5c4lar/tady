cmake_minimum_required(VERSION 3.25)

project(tady)

set(CMAKE_CXX_STANDARD 20)

# Set installation directory handling
if(SKBUILD_CORE OR SKBUILD)
  # scikit-build-core sets up CMAKE_INSTALL_PREFIX to point to the correct site-packages directory
  message(STATUS "Building with scikit-build-core")
else()
  # Default installation location when not building with scikit-build
  set(PYTHON_INSTALL_DIR "lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/site-packages/tady")
endif()

# Add the C++ subdirectory
add_subdirectory(cpp)
