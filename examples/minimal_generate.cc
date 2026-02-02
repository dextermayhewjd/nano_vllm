#include <iostream>
#include "llm/api/engine.h"

int main(){
    llm::api::Engine engine;
    std::cout<<"S00 smoke test: build works.\n"<< engine.Ping()<<std::endl;
    return 0;
}
