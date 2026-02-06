cmake_minimum_required(VERSION 3.20)
project(llm_inference_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(minimal_generate)

target_sources(minimal_generate
PRIVATE
examples/minimal_generate.cc
)

add_library(llm_engine)
target_sources(llm_engine
  PRIVATE
    src/api/engine.cc
    src/utils/logging.cc 
)

target_include_directories(llm_engine
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include 
)

target_link_libraries(minimal_generate PRIVATE llm_engine)

