cmake_minimum_required(VERSION 3.28.3)
project(llm_inference_engine LANGUAGES CXX)

# 选用c++20 
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(minimal_generate)

target_sources(minimal_generate
  PRIVATE
    examples/minimal_generate.cc
)