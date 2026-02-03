#include "llm/api/engine.h"
#include "llm/utils/status.h"
#include <memory>

namespace llm::api {

    Engine::Engine() = default;
    llm::utils::StatusOr<std::unique_ptr<Engine>>Engine::Create(bool simulate_failure)
    {
        if(simulate_failure)
        {
            return llm::utils::StatusOr<std::unique_ptr<Engine>>{
                llm::utils::Status("simulated init failure")
            };
        }   
        return std::unique_ptr<Engine>(new Engine());
    }
    
    Engine::~Engine() = default;

    llm::utils::StatusOr<const char*> Engine::Ping()const{
        return "pong";
    }

} // namespace llm::api