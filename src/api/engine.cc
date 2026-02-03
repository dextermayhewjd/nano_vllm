#include "llm/api/engine.h"
#include "llm/utils/status.h"
#include <memory>

namespace llm::api {

    Engine::Engine(std::string model_path)
        :model_path_(std::move(model_path)){}

    llm::utils::StatusOr<std::unique_ptr<Engine>>Engine::Create(EngineConfig cfg)
    {
        if(cfg.model_path.empty()) // 这里是string的function吗 是的 这里是用来检查是否满足成功条件
        {
            return llm::utils::StatusOr<std::unique_ptr<Engine>>{
                llm::utils::Status("model_path is empty")
            };
        }   
        return std::unique_ptr<Engine>(new Engine(std::move(cfg.model_path)));
    }
    
    Engine::~Engine() = default;

    llm::utils::StatusOr<const char*> Engine::Ping()const{
        return "pong";
    }

} // namespace llm::api