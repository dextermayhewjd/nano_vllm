// include/llm/api/request.h
#pragma once
#include <string>

#include "llm/api/types.h"  // 用到 SamplingParams，所以这里 include types.h

namespace llm::api{

    struct GenerateRequest
    {
        std::string prompt;
        SamplingParams sampling{};   
        // 关键：请求里带采样参数（默认值来自 types.h）
    };


} // namespace llm ::api
