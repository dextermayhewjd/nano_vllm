#include "llm/api/engine.h"
#include "llm/utils/status.h"
#include <memory>

namespace llm::api {

    Engine::Engine() = default;
    llm::utils::StatusOr<std::unique_ptr<Engine>>Engine::Create()
    {
        return std::unique_ptr<Engine>(new Engine());
    }
    
    Engine::~Engine() = default;

    llm::utils::StatusOr<const char*> Engine::Ping()const{
        return "pong";
    }

} // namespace llm::api