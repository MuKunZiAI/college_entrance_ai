from langchain_ollama import ChatOllama
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool

# 1. 定义工具（示例）
def dummy_tool(query: str) -> str:
    return f"搜索结果：{query}"

tools = [
    Tool(
        name="Search",
        func=dummy_tool,
        description="用于搜索信息"
    )
]

# 2. 加载 ReAct 提示模板
prompt = hub.pull("hwchase17/react")

# 3. 创建 LLM
llm = ChatOllama(
    model="qwen:7b",
    temperature=0.7,
    base_url="http://localhost:11434"
)

# 4. 创建 Agent 和执行器
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)

# 5. 调用
response = agent_executor.invoke({"input": "今天北京天气怎么样？"})
print(response["output"])