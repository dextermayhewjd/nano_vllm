#include <iostream>
#include "llm/api/engine.h"

int main()
{
    llm::api::Engine engine;
    
    auto status = engine.Ping();

    if(!status.ok())
    {
        std::cerr<<"Ping failed:" <<status.message()<<"\n";

    }
    std::cout << "S00 step6: Engine::Ping() ok\n";
  return 0;
}
