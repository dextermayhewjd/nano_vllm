#include <iostream>
#include "llm/api/engine.h"

int main(){

    // ===== 1) 先验证失败路径：model_path 为空应当失败 =====
    {
        llm::api::EngineConfig bad_cfg;
        bad_cfg.model_path = "";  // 空字符串 -> 期望 Create 失败

        auto bad_or = llm::api::Engine::Create(bad_cfg);
        if (bad_or.ok()) {
        // 这一行这会应该不运行
        std::cerr << "Unexpected: Create(empty model_path) succeeded\n";
        return 1;
        }
        std::cerr << "Expected failure: " << bad_or.status().message() << "\n";
    }

        // ===== 2) 再验证成功路径：model_path 非空应当成功 =====
    llm::api::EngineConfig cfg;
    cfg.model_path = "dummy_model_path";  // Phase0 只是占位字符串

    auto engine_or = llm::api::Engine::Create(cfg);
    if (!engine_or.ok()) {
        // 这里不应该运行
        std::cerr << "Create failed: " << engine_or.status().message() << "\n";
        return 1;
    }

    // 注意：unique_ptr 只能 move，所以要用 std::move(engine_or).value()
    auto engine = std::move(engine_or).value();


    llm::api::GenerateRequest req;
    req.prompt = "Hello from S00!";
    req.sampling.max_new_tokens = 16;
    req.sampling.temperature = 0.8f;

    auto gen_or = engine->Generate(req);
    if (!gen_or.ok()) {
        std::cerr << "Generate failed: " << gen_or.status().message() << "\n";
    return 1;
    }

    std::cout << gen_or.value().text << "\n";


    // ===== 3) 调用一个最小方法，验证“调用 + 返回值 + StatusOr”链路 =====
    auto ping_or = engine->Ping();
    if (!ping_or.ok()) {
        std::cerr << "Ping failed: " << ping_or.status().message() << "\n";
        return 1;
    }

    std::cout << "Phase0 minimal_generate: Create ok, Ping -> "
                << ping_or.value() << "\n";

    return 0;
}
