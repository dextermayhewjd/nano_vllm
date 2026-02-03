#pragma once
#include <memory>
#include "llm/utils/status.h"
namespace llm::api{


    class Engine{
    
    public:   
        // 构造不再直接暴露
        static llm::utils::StatusOr<std::unique_ptr<Engine>> Create();   
        ~Engine();

        llm::utils::StatusOr<const char*> Ping()const;

    private:
        Engine(); // 构造函数是 private：只能 Create 内部用
    };

} // namespace llm:api