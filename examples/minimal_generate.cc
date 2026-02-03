#include <iostream>
#include "llm/api/engine.h"

int main(){
    auto engine_or = llm::api::Engine::Create();

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
