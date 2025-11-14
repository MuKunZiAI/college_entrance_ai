from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from langchain_ollama import ChatOllama

# 1. 定义工具（示例）
def dummy_tool(query: str) -> str:
    return f"搜索结果：北京今天天气晴，气温25°C，东南风2级，空气质量良好。"

tools = [
    Tool(
        name="Search",
        func=dummy_tool,
        description="用于搜索信息"
    )
]

# 2. 加载 ReAct 提示模板
prompt = PromptTemplate.from_template("""
你是一个智能助手，能够调用以下工具：
{tools}

工具名称是：{tool_names}之一

回答用户问题: {input}

{agent_scratchpad}
""")

# 3. 创建 LLM
llm = ChatOllama(
    model="qwen3:32b",
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