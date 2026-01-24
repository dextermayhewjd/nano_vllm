# include "llm/api/engine.h"

namespace llm::api {
    Engine::Engine() = default;
    Engine::~Engine() = default;

    llm::utils::StatusOr<const char*> Engine::Ping() const{
        return "pong"; //pong 这里"pong" → const char* 自动构造
        // 永远成功
    }
}// namespace llm::api