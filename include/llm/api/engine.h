#pragma once
#include <memory>
#include "llm/utils/status.h"
#include "llm/api/types.h" 
#include "llm/api/request.h"

namespace llm::api{


    class Engine{
    
    public:   
        // 构造不再直接暴露
        static llm::utils::StatusOr<std::unique_ptr<Engine>> Create(EngineConfig cfg);   
        ~Engine();

        llm::utils::StatusOr<const char*> Ping()const;

        llm::utils::StatusOr<GenerateResult> Generate(const GenerateRequest& req) const;

    private:
        explicit Engine(std::string model_path);
        std::string model_path_;
    };

} // namespace llm:api