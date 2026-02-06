# llm-inference-engine

一个从 0 开始用现代 C++ 搭建的 LLM 推理引擎项目（学习/工程化取向）。  
当前处于 **S00：最小可编译构建（Build + API 空壳）** 阶段。

> S00 目标：  
> - 项目能 **编译 + 链接 + 运行**  
> - 有一套最小的 **API 形状**（Create / Ping / Generate stub）  
> - 有最小的 **错误流**（Status / StatusOr）  
> - 目录结构与后续阶段可扩展

---

## 目录结构（S00）

```text
llm-inference-engine/
├── CMakeLists.txt
├── include/
│   └── llm/
│       ├── api/
│       │   ├── engine.h
│       │   ├── request.h
│       │   └── types.h
│       └── utils/
│           ├── logging.h
│           └── status.h
├── src/
│   ├── api/
│   │   └── engine.cc
│   └── utils/
│       └── logging.cc
└── examples/
    └── minimal_generate.cc
```

## 构建与运行
依赖  
CMake >= 3.20
C++ 编译器支持 C++20（clang / gcc 均可）

### build 指令

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j
```

### Run example

```bash
./build/minimal_generate
```

预期行为（大致）：

- Create(empty model_path) 走失败分支，打印错误信息
- Create(ok) 成功后：
  - Ping() 返回 "pong"
  - Generate() 返回一段 stub 文本（包含 prompt 与 sampling 参数）

- Engine 内部会通过最小 logger 输出少量 INFO/WARN 日志（走 stderr）

## 当前 API（S00）

### 错误流：Status / StatusOr
  - llm::utils::Status 表示成功/失败（失败带 message）
  - llm::utils::StatusOr< T > 表示：  
        成功：持有一个 T  
        失败：持有一个 Status  

- 注意：失败构造 StatusOr(Status) 是 explicit，失败返回时需要显式构造 StatusOr<T>{ Status("...") }。

### Engine

- Engine::Create(EngineConfig cfg) -> StatusOr<std::unique_ptr<Engine>>

- cfg.model_path 为空：失败
- 非空：成功（当前仅保存 model_path，不加载模型）

- Engine::Ping() -> StatusOr<const char*>  
  - stub：返回 "pong"  
- Engine::Generate(const GenerateRequest&) -> StatusOr<GenerateResult>

- prompt 为空：失败
- 非空：返回 stub 文本（不做真实推理）

### Request / Types

- EngineConfig：目前只有 model_path  
- SamplingParams：目前只有 max_new_tokens / temperature  
- GenerateRequest：prompt + sampling  
- GenerateResult：目前只有 text  