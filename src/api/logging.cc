#include <atomic>
#include <iostream>
#include <string_view>
#include "llm/utils/logging.h"
namespace llm::utils 
{

    namespace
    {
        std::atomic<LogLevel> g_level{LogLevel::kInfo};
    
        const char* LevelToString(LogLevel level)
        {
            switch (level)
            {
                case llm::utils::LogLevel::kDebug: return "DEBUG";
                case llm::utils::LogLevel::kInfo: return "INFO";
                case llm::utils::LogLevel::kWarn: return "WARN";
                case llm::utils::LogLevel::kError: return "ERROR";
            }
            // switch case 的语法我是知道的
            return "UNKNOWN";
        }
    }//namespace

    void SetLogLevel(LogLevel level)
    {
        g_level.store(level,std::memory_order_relaxed);
    }

    LogLevel GetLogLevel()
    {
        return g_level.load(std::memory_order_relaxed);
    }

    void log(LogLevel level, std::string_view msg)
    {
          // 级别过滤：level < 当前阈值 => 不输出
        if (static_cast<int>(level) < static_cast<int>(GetLogLevel()))
        {
            return;
        }
          // S00：先全部输出到 stderr，避免 stdout 和程序输出混在一起不好读
        std::cerr << "[" << LevelToString(level) << "] " << msg << "\n";
    }


} // namespace::utils