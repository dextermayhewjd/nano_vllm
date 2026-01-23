# include "llm/api/engine.h"

namespace llm::api {
    Engine::Engine() = default;
    Engine::~Engine() = default;

    const char* Engine::Ping() const{
        return "pong";
    }
}