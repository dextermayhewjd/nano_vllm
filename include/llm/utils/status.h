#pragma once
#include <string>

namespace llm::utils 
{

    // S00/S01 阶段：极简 Status
    class Status 
    {
        public:
        // 成功态
        Status() = default;

        // 失败态
        explicit Status(std::string msg)
            : ok_(false), message_(std::move(msg)) {}

        bool ok() const { return ok_; }
        const std::string& message() const { return message_; }

    private:
        bool ok_{true};
        std::string message_;
    };


/*
为了支持在loadvalue的时候的状态值设置的模板状态机 
*/

// ===== 新增：StatusOr<T> =====
template <typename T>
    class StatusOr
    {
    
    public:
    // 构造失败态
        StatusOr(Status status)
        :status_(std::move(status)),has_value_(false){}
    // 构造成功态    
        StatusOr(T value)
        :status_(),value_(std::move(value)),has_value_(true){}
        
        // bool ok不ok
        bool ok()const {return status_.ok();}
        const Status& status(){return status_;}

        // 取值（ 1. 左值 2. const 加左值 3. 右值）
        T& value() & { return value_; }
        const T& value() const & { return value_; }
        T&& value() && { return std::move(value_); }
    private:
        Status status_;
        T value_;
        bool has_value_{false};
    };

} // namespace llm::utils