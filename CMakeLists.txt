cmake_minimum_required(VERSION 3.10)
project(KrishLLM)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Configure llama.cpp BUILD options (disable tests, examples, etc to speed up build)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "Build tests" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "Build examples" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "Build server" FORCE)

# Add llama.cpp as a subdirectory
add_subdirectory(llama.cpp)

# Optimize for performance for our wrapper
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -funroll-loops")

add_executable(KrishLLM main.cpp)

# Link the llama backend libraries
target_link_libraries(KrishLLM PRIVATE llama ggml)

# Include the headers from llama.cpp
target_include_directories(KrishLLM PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/include 
    ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/ggml/include
)

