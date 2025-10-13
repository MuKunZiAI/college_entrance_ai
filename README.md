# college_entrance_ai
高考信息查询智能助手

## 本地环境

| 工具名称        | 推荐版本              | 说明                        |
|-----------------|-------------------|---------------------------|
| 操作系统        | macOS 15.6.1      | MacBook pro m1 max 32GB   |
| Docker Desktop  | 28.3.2            | 本地Docker环境                |
| Python          | 3.13              | python3.13版本实现我们的AI Agent |
| IDE/编辑器      | VS Code / PyCharm | 开发工具                      |

## 运行环境

| 服务名称     | 版本   | 端口  | 说明                                          |
|--------------|--------|-------|-----------------------------------------------|
| Ollama       | 0.11.6 | 11434 | 大模型运行环境，运行deepseek-r1:32b、qwen3:32b、llama2 |
| MySQL        | 8.4.6  | 3306  | 本地docker部署，'业务知识库'                   |
| ElasticSearch| 8.19.0 | 9200  | 本地docker部署，支持向量存储、检索              |

## 代码结构

```python
college_entrance_ai/
├── agentscope_code/             # `AgentScope框架`实现
│   └── agent_service_react.py   # ReAct模式智能体
├── langchain_code/              # `LangChain框架`实现
│   ├── agent_service_react.py   # ReAct模式智能体
│   └── agent_service_seq.py     # SequentialChain模式智能体
├── manual_code/                 # `手写代码`实现
│   └── agent_service.py         # 手写代码模式智能体
├── qwenagent_code/              # `Qwen-Agent框架`实现
│   └── agent_service_react.py   # ReAct模式智能体
├── api_service.py               # API服务，连接MySQL，Elasticsearch8，调用本地大模型
└── README.md                    # 说明
```
