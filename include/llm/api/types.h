#pragma once
#include <string>

namespace llm::api {

    struct EngineConfig 
    {
    std::string model_path;
    };
        
    // 最小采样参数（只放最常见的两个）
    struct SamplingParams {
    int max_new_tokens{32};
    float temperature{1.0f};
    };

    // 最小生成结果（只返回文本）
    struct GenerateResult {
    std::string text;
    };
} // namespace llm::api
