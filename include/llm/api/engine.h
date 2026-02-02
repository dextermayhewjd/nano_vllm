#pragma once

#include "llm/utils/status.h"
namespace llm::api{


    class Engine{
    
    public:   
        Engine();
        ~Engine();

        llm::utils::Status Ping()const;
    };

} // namespace llm:api