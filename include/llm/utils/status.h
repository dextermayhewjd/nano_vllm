#pragma once
#include <cassert>
#include <string>
#include <optional>
#include <utility>   // ✅ 为 std::move 提供声明
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
        explicit StatusOr(Status status)
        :status_(std::move(status)){
            if(status_.ok())
            {
            status_ = Status("StatusOr(Status) requires non-ok status");
            assert(false && "StatusOr(Status) requires non-ok status");           
            }
        }
    // 构造成功态    
        StatusOr(T value)
        :status_(),value_(std::move(value)){}
        

        // bool ok不ok
        bool ok()const {return value_.has_value();}
        const Status& status()const{return status_;}

        // 取值（ 1. 左值 2. const 加左值 3. 右值）
        T& value() & {
            assert(ok());
            return *value_;
        }
        const T& value() const & {
            assert(ok());
            return *value_;
        }
        T&& value() && {
            assert(ok());
            return std::move(*value_);
        }
    private:
        Status status_;
        std::optional<T> value_;
    };

} // namespace llm::utils