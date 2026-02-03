#pragma once
#include <memory>
#include "llm/utils/status.h"
namespace llm::api{


    class Engine{
    
    public:   
        // 构造不再直接暴露
        static llm::utils::StatusOr<std::unique_ptr<Engine>> Create(std::string model_path);   
        ~Engine();

        llm::utils::StatusOr<const char*> Ping()const;

    private:
        explicit Engine(std::string model_path);
        std::string model_path_;
    };

} // namespace llm:api