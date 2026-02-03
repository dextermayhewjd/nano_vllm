#pragma once
#include <string_view>

namespace llm::utils {

// S00：最小日志级别
enum class LogLevel {
  kDebug = 0,
  kInfo  = 1,
  kWarn  = 2,
  kError = 3,
};

// 设置/获取全局日志级别（S00 先用全局，后面再做 logger 实例化也不迟）
void SetLogLevel(LogLevel level);
LogLevel GetLogLevel();

// 核心日志接口：根据 level 输出一条消息
void Log(LogLevel level, std::string_view msg);

// 便捷封装：让调用点更干净
inline void Debug(std::string_view msg) { Log(LogLevel::kDebug, msg); }
inline void Info(std::string_view msg)  { Log(LogLevel::kInfo,  msg); }
inline void Warn(std::string_view msg)  { Log(LogLevel::kWarn,  msg); }
inline void Error(std::string_view msg) { Log(LogLevel::kError, msg); }

} // namespace llm::utils
