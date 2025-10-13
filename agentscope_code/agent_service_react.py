import json
import asyncio

from agentscope.message import TextBlock, Msg
from agentscope.tool import ToolResponse, Toolkit
from agentscope.agent import AgentBase, ReActAgent
from agentscope.formatter import OllamaChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.model import OllamaChatModel

from api_service import QueryService, SemanticService, AnalysisService

# 初始化服务
queryService = QueryService()
semanticService = SemanticService()
analysisService = AnalysisService()


# 自定义工具类
# 根据输入文本语义匹配表结构，每次返回一个最相关的表结构。对于需要多表查询的问题，需要多次调用此工具。
async def match_metadata(user_query: str) -> ToolResponse:
    table = semanticService.hybrid_search(user_query, 1)

    return ToolResponse(
        content=[
            TextBlock(
                type="text",
                text=f"{[t['table_info'] for t in table]}",
            ),
        ],
    )


# 根据输入文本语义匹配表结构，每次返回一个最相关的表结构。对于需要多表查询的问题，需要多次调用此工具。
async def execute_sql(query_sql: str) -> ToolResponse:
    result = json.dumps(queryService.query_with_column(query_sql))

    return ToolResponse(
        content=[
            TextBlock(
                type="text",
                text=f"{result}",
            ),
        ],
    )


async def interactive_react_agent() -> None:
    """创建一个支持多轮对话的ReAct智能体。"""
    # 准备工具
    toolkit = Toolkit()
    toolkit.register_tool_function(match_metadata)
    toolkit.register_tool_function(execute_sql)

    jarvis = ReActAgent(
        name="Jarvis",
        sys_prompt="""
            你是一个数据分析助手，负责帮助用户查询数据库信息。
            请特别注意：用户的问题可能需要从多个表中查询数据。
            1. 首先确定需要查询哪些数据
            2. 使用match_metadata工具分别匹配包含这些数据的表结构
            3. 对每个表生成相应的SQL查询语句
            4. 执行查询并汇总结果
            5. 最后计算并给出答案

            请确保逐步执行，不要跳过任何步骤。
        """,
        model=OllamaChatModel(
            model_name="qwen3:32b",  # 指定模型名称
            stream=True,  # 根据需要设置是否流式输出
            enable_thinking=True,  # 为Qwen3启用思考功能（可选）
            # host="http://localhost:11434" # 如果Ollama不在默认地址，需指定
        ),
        formatter=OllamaChatFormatter(),
        toolkit=toolkit,
        memory=InMemoryMemory(),
    )

    print("数据分析助手已启动！输入问题进行查询（输入'exit'退出）")
    print("-" * 50)

    while True:
        # 使用run_in_executor处理阻塞的input调用
        user_input = await asyncio.get_event_loop().run_in_executor(
            None,
            input,
            "用户: "
        )

        # 检查是否退出指令 [[1]]
        if user_input.strip().lower() == 'exit':
            print("正在退出程序...")
            break

        if not user_input.strip():
            continue

        # 创建消息并发送给智能体
        msg = Msg(
            name="user",
            content=user_input,
            role="user",
        )

        print("\nJarvis正在思考...")

        # 获取智能体响应
        response = await jarvis(msg)

        # 显示回答
        print(f"\nJarvis: {response.content}\n")
        print("-" * 50)


# 运行交互式智能体
if __name__ == "__main__":
    asyncio.run(interactive_react_agent())