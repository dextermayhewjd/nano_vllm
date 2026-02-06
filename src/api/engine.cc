#include "llm/api/engine.h"
#include "llm/api/request.h"
#include "llm/api/types.h"
#include "llm/utils/logging.h"
#include "llm/utils/status.h"
#include <memory>
#include <sstream>

namespace llm::api {
    
    /*
    对内的调用
    */
    Engine::Engine(std::string model_path)
        :model_path_(std::move(model_path)){}


    /*
    对外的创建Engine api
    */
    llm::utils::StatusOr<std::unique_ptr<Engine>>Engine::Create(EngineConfig cfg)
    {
        if(cfg.model_path.empty()) // 这里是string的function 是用来检查是否满足成功条件
        {
            llm::utils::Warn("Engine::Create 失败 model_path 是空的");
            return llm::utils::StatusOr<std::unique_ptr<Engine>>{
                llm::utils::Status("model_path is empty")
            };
        }   

        llm::utils::Info("Engine::Create 成功 ");
        return std::unique_ptr<Engine>(new Engine(std::move(cfg.model_path)));
    }
    
    /*
    析构Engine 
    */
    Engine::~Engine() = default;


    llm::utils::StatusOr<const char*> Engine::Ping()const
    {
        return "pong";
    }


    /*基于request创建 result
    所有有问题的过程 都靠着 
    1. 检查输入本身 -> 
    2.要么生成status  （ 注意因为explicit原因 失败状态需要 显示构造 -> 
    3.正常返回的值直接包转化为statusOR< 里面 >
    */
    llm::utils::StatusOr<llm::api::GenerateResult>
    Engine::Generate(const llm::api::GenerateRequest& req)const
    {
        if(req.prompt.empty())
        {
            llm::utils::Warn("Engine:: Generate 失败 prompt本身是空的");
            return llm::utils::StatusOr<llm::api::GenerateResult>{
                llm::utils::Status("GenerateRequest.prompt is empty")
            };
        }
    
    llm::utils::Info("Engine::Generate ok (开始输出stub)");
    std::ostringstream oss;
    oss << "=== S00 stub generate ===\n"
        << "prompt: " << req.prompt << "\n"
        << "max_new_tokens: " << req.sampling.max_new_tokens << "\n"
        << "temperature: " << req.sampling.temperature << "\n";

    GenerateResult out;
    out.text = oss.str();
    return out; // 成功态（当前 StatusOr(T) 不是 explicit，所以可以直接 return out）
    }


} // namespace llm::api