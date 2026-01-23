#include <iostream>
#include "llm/api/engine.h"

int main()
{
    llm::api::Engine engine;
    std::cout << "S00 step5: Engine::Ping() -> " << engine.Ping() << "\n";
}
