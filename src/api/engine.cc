# include "llm/api/engine.h"

namespace llm::api {
    Engine::Engine() = default;
    Engine::~Engine() = default;

    llm::utils::Status Engine::Ping() const{
        return llm::utils::Status{};
        // 永远成功
    }
}