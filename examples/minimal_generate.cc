#include <iostream>
#include "llm/api/engine.h"

int main(){
    llm::api::Engine engine;

    auto status = engine.Ping();
    if(!status.ok())
    {
        std::cerr << "Ping failed: " << status.message() << "\n";
        return 1;
    }

    std::cout<<"S00 smoke test: engine::ping() ok\n"
             <<std::endl;
    return 0;
}
