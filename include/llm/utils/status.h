#pragma once
#include <cassert>
#include <optional>
#include <string>

namespace llm::utils {

    class Status
    {
        public:
        //成功态
            Status() = default;
        //失败态
        explicit Status(std::string msg):
                        ok_(false),
                        message_(std::move(msg)){}

        bool ok()const {return ok_;}
        const std::string& message()const{return message_;}

        private:
            bool ok_{true};
            std::string message_;
    };

    template <typename T>
    class StatusOr
    {
        public:
          //  失败态度
          explicit StatusOr(Status status):status_(std::move(status))
          {
            if (status_.ok()) 
            {
                status_ = Status("StatusOr(Status) requires non-ok status");
                assert(false && "StatusOr(Status) requires non-ok status");
            }
          }
          
          // 成功态
          StatusOr(T value)
          :value_(std::move(value)){}

          // getter method
          
          bool ok()const
          {
            return value_.has_value();
          }

          const Status& status()const
          {
            return status_;
          }

        T& value() &
        {
            assert(ok());
            return *value_;
        }

        const T& value()const &
        {
            assert(ok());
            return *value_;
        }

        T&& value() &&
        {
            assert(ok());
            return std::move(*value_);
        }

        private:
            Status status_;
            std::optional<T> value_;

    };



}// namespace llm::utils

