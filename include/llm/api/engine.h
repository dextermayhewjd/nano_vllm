#pragma once

namespace llm::api{


    class Engine{
    
    public:   
        Engine();
        ~Engine();

        const char* Ping()const;
    };

} // namespace llm:api