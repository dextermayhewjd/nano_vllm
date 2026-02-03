#include <iostream>
#include "llm/api/engine.h"

int main(){
        
    // 1) 先故意触发一次失败：验证错误链路
    {
    auto bad_or = llm::api::Engine::Create("");
    if (bad_or.ok()) {
        std::cerr <<  "Unexpected: Create(\"\") succeeded\n";
        return 1;
    }
    std::cerr << "Expected failure: " << bad_or.status().message() << "\n";
    }

    auto engine_or = llm::api::Engine::Create("dummy_model_path");

    if(!engine_or.ok())
    {
        std::cerr << "Create failed: " << engine_or.status().message() << "\n";
        return 1;
    }

    auto engine = std::move(engine_or).value();
     // 拿到 std::unique_ptr<Engine>
    auto ping_or = engine->Ping();
    if (!ping_or.ok()) 
    {
        std::cerr << "Ping failed: " << ping_or.status().message() << "\n";
    return 1;
    }   

    std::cout << "S00 step8: Engine::Create ok, Ping -> " << ping_or.value() << "\n";
    return 0;
}
