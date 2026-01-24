# =========================
# CMake 基础元信息
# =========================

# 指定最低 CMake 版本
# 含义：保证后面使用的 target_* 现代范式全部可用
cmake_minimum_required(VERSION 3.28.3)
# 定义整个工程
# LANGUAGES CXX 表示：这是一个 C++ 工程（不会隐式启用 C）
project(llm_inference_engine LANGUAGES CXX)

# 指定使用 C++20
# 这是“编译语义”，最终会变成 -std=c++20
set(CMAKE_CXX_STANDARD 20)
# 强制要求编译器支持 C++20
# 否则 CMake 配置阶段直接失败（而不是偷偷降级）
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 开给vscode看
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# =========================
# 可执行程序 target
# =========================

# 定义一个可执行 target，但暂时不指定源文件
# 这一步只是“声明 target 存在”
add_executable(minimal_generate)

# 给 minimal_generate 这个 target 添加源文件
# target_sources 是现代写法：源文件是 target 的属性
target_sources(minimal_generate
  PRIVATE
    examples/minimal_generate.cc
)
# =========================
# 核心库 target（llm_engine）
# =========================

# 定义一个 library target
# 含义：把 engine.cc 编译成一个可被链接的库（.a / .so）
add_library(llm_engine 
    src/api/engine.cc
)
# =========================
# target 之间的“实现依赖”（link）
# =========================
# 表示：
# minimal_generate 在“链接阶段”依赖 llm_engine
# 即：
# - minimal_generate 会用到 llm_engine 里实现的函数
# - llm_engine 的二进制产物会被链接进最终可执行文件
target_link_libraries(minimal_generate PRIVATE llm_engine)

# =========================
# llm_engine 的 API 可见性（include）
# =========================

# 声明 llm_engine 的“公共头文件”所在位置
#
# PUBLIC 的含义是：
# 1. llm_engine 自己在编译 engine.cc 时，需要 include/ 下的头文件
# 2. 所有 link 了 llm_engine 的 target（如 minimal_generate）
#    在编译时，也自动获得这个 include 路径
#
# 这一步解决的是：
#   #include "llm/api/engine.h"
#   编译器“去哪找这个头文件”的问题
target_include_directories(llm_engine
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
# test 相关的 同时也是option
option(LLM_ENABLE_TESTS "Build unit tests" ON)


# ===== Subdirectories =====

# 加入 tests 子目录
if (LLM_ENABLE_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()