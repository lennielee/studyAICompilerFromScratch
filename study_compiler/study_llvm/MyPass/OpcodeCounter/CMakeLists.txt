cmake_minimum_required(VERSION 3.20)
project(llvm-tutor-opcode-counter)

#===============================================================================
# 1. LOAD LLVM CONFIGURATION
#===============================================================================
# Set this to a valid LLVM installation dir
set(LT_LLVM_INSTALL_DIR "" CACHE PATH "LLVM installation directory")

# Add the location of LLVMConfig.cmake to CMake search paths (so that find_package can locate it)
list(APPEND CMAKE_PREFIX_PATH "${LT_LLVM_INSTALL_DIR}/lib/cmake/llvm/")

find_package(LLVM CONFIG)
if("${LLVM_VERSION_MAJOR}" VERSION_LESS 19)
  message(FATAL_ERROR "Found LLVM ${LLVM_VERSION_MAJOR}, but need LLVM 19 or above")
endif()

# Add LLVM include directories and your own header directory
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS} inc)

#===============================================================================
# 2. LLVM-TUTOR BUILD CONFIGURATION
#===============================================================================
# Use the same C++ standard as LLVM does
set(CMAKE_CXX_STANDARD 17 CACHE STRING "")

# LLVM is normally built without RTTI. Be consistent with that.
if(NOT LLVM_ENABLE_RTTI)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")
endif()

#===============================================================================
# 3. ADD THE TARGET
#===============================================================================
# Build the OpcodeCounter plugin using the source file in src/OpcodeCounter.cpp.
add_library(OpcodeCounter SHARED src/OpcodeCounter.cpp)

# Allow undefined symbols in shared objects on Darwin (default behavior on Linux)
target_link_libraries(OpcodeCounter
  "$<$<PLATFORM_ID:Darwin>:-undefined dynamic_lookup>"
)
