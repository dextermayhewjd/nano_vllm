#include <iostream>
#include "llm/api/engine.h"

int main()
{
    llm::api::Engine engine;
    
    auto result = engine.Ping();
    // 失败态获取信息
    if(!result.ok())
    {
        std::cerr<<"Ping failed:" <<result.status().message()<<"\n";
        return 1;
    }

    // 成功态获取信息
    std::cout << "S00 step7: Engine::Ping() -> "
            << result.value() << "\n";
  return 0;
}
