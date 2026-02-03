#include "llm/api/engine.h"
#include "llm/api/request.h"
#include "llm/api/types.h"
#include "llm/utils/status.h"
#include <memory>
#include <sstream>

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

    llm::utils::StatusOr<const char*> Engine::Ping()const
    {
        return "pong";
    }


    llm::utils::StatusOr<llm::api::GenerateResult>
    Engine::Generate(const llm::api::GenerateRequest& req)const
    {
        if(req.prompt.empty())
        {
            return llm::utils::StatusOr<llm::api::GenerateResult>{
                llm::utils::Status("GenerateRequest.prompt is empty")
            };
        }

    std::ostringstream oss;
    oss << "=== S00 stub generate ===\n"
        << "prompt: " << req.prompt << "\n"
        << "max_new_tokens: " << req.sampling.max_new_tokens << "\n"
        << "temperature: " << req.sampling.temperature << "\n";

    GenerateResult out;
    out.text = oss.str();
    return out; // 成功态（当前 StatusOr(T) 不是 explicit，所以可以直接 return out）
    }

} // namespace llm::api