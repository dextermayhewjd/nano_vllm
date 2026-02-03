#include <iostream>
#include <utility>

#include "llm/api/engine.h"
#include "llm/api/request.h"
#include "llm/api/types.h"

int main() {
  // ===== 1) Create 失败路径：空 model_path =====
  {
    llm::api::EngineConfig bad_cfg;
    bad_cfg.model_path = "";

    auto bad_or = llm::api::Engine::Create(bad_cfg);
    if (bad_or.ok()) {
      std::cerr << "Unexpected: Create(empty model_path) succeeded\n";
      return 1;
    }
    std::cerr << "Expected Create failure: " << bad_or.status().message() << "\n";
  }

  // ===== 2) Create 成功路径 =====
  llm::api::EngineConfig cfg;
  cfg.model_path = "dummy_model_path";

  auto engine_or = llm::api::Engine::Create(cfg);
  if (!engine_or.ok()) {
    std::cerr << "Create failed: " << engine_or.status().message() << "\n";
    return 1;
  }

  auto engine = std::move(engine_or).value();

  // ===== 3) Ping 验证 =====
  auto ping_or = engine->Ping();
  if (!ping_or.ok()) {
    std::cerr << "Ping failed: " << ping_or.status().message() << "\n";
    return 1;
  }
  std::cout << "S00: Ping -> " << ping_or.value() << "\n";

  // ===== 4) Generate 失败路径：空 prompt =====
  {
    llm::api::GenerateRequest bad_req;
    bad_req.prompt = "";
    bad_req.sampling.max_new_tokens = 8;
    bad_req.sampling.temperature = 1.0f;

    auto bad_gen_or = engine->Generate(bad_req);
    if (bad_gen_or.ok()) {
      std::cerr << "Unexpected: Generate(empty prompt) succeeded\n";
      return 1;
    }
    std::cerr << "Expected Generate failure: " << bad_gen_or.status().message() << "\n";
  }

  // ===== 5) Generate 成功路径 =====
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
  return 0;
}
