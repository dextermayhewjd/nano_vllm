#include "llm/api/engine.h"
#include "llm/utils/status.h"

namespace llm::api {

    Engine::Engine() = default;
    Engine::~Engine() = default;

    llm::utils::Status Engine::Ping()const{
        return llm::utils::Status{};
    }

} // namespace llm::api