#pragma once

# include "llm/utils/status.h"

namespace llm::api {

// S00 阶段：空壳类，只证明“API 形态 + include 路径”
// 不放任何方法、不放构造函数、不放成员
    class Engine 
    {
    public:
        Engine();
        ~Engine();
        // 加上构造函数和析构函数

        llm::utils::StatusOr<const char*> Ping() const;
    };

} // namespace llm::api
