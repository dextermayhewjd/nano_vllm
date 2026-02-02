#pragma once
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

}// namespace llm::utils