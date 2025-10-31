import asyncio
import json

from qwen_agent.agents import Assistant
from qwen_agent.tools import BaseTool

from api_service import QueryService, SemanticService, AnalysisService
from message_manage import MessageManager
from qwenagent_api import read_steam_response

# 初始化服务
queryService = QueryService()
semanticService = SemanticService()
analysisService = AnalysisService()
# 初始化管理器，最多保留 10 轮对话（即 20 条消息）
mm = MessageManager(max_history=10)
session_id = "uuid1"


# 自定义工具类
class MatchMetadataTool(BaseTool):
    name = 'match_metadata'
    description = '根据输入文本语义匹配表结构，每次返回一个最相关的表结构。对于需要多表查询的问题，需要多次调用此工具。'

    def call(self, params, **kwargs) -> str:
        table = semanticService.hybrid_search(params, 1)
        return f"{[t['table_info'] for t in table]}"


class ExecuteSQLTool(BaseTool):
    name = 'execute_sql'
    description = '执行SQL查询并返回结果。输入应为标准SQL语句。注意：可能需要执行多个SQL查询来获取不同表中的数据。'

    def call(self, params, **kwargs) -> str:
        if isinstance(params, str):
            jsonObj = json.loads(params)
            if 'sql' in jsonObj:
                params = jsonObj['sql']
            if 'query' in jsonObj:
                params = jsonObj['query']
        if isinstance(params, dict):
            if 'sql' in params:
                params = params['sql']
            if 'query' in params:
                params = params['query']
        if params.endswith(';'):
            params = params[:-1]
        return json.dumps(queryService.query_with_column(params))

# 创建Agent实例
agent = Assistant(
    name='ai_agent_assistant',
    llm={
        'model': 'qwen3:32b',
        'model_server': 'http://localhost:11434/v1',
    },
    system_message="""
        你是一个数据分析助手，负责帮助用户查询数据库信息。
        请特别注意：用户的问题可能需要从多个表中查询数据。
        1. 首先确定需要查询哪些数据
        2. 使用match_metadata工具分别匹配包含这些数据的表结构
        3. 对每个表生成相应的SQL查询语句
        4. 执行查询并汇总结果
        5. 最后计算并给出答案

        请确保逐步执行，不要跳过任何步骤。
    """,
    function_list=[MatchMetadataTool(), ExecuteSQLTool()],
)

def analysis(user_query):
    try:
        # 运行Agent
        messages = mm.get_messages(session_id) + [{'role': 'user', 'content': user_query}]
        response_generator = agent.run(messages=messages)
        # 处理生成器响应
        full_response = read_steam_response(response_generator)
        # 缓存历史对话
        if full_response:
            mm.add_user_message(session_id, user_query)
            mm.add_assistant_message(session_id, full_response)
        return full_response
    except Exception as e:
        print(f"执行过程中出错: {str(e)}")
        # 这里可以添加重试或更详细的错误处理逻辑
        return f"错误: {str(e)}"

# agent 智能体核心流程
async def chat():
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

        print("\nJarvis正在思考...")
        # 调用
        analysis(user_input)
        print("\n")
        print("-" * 50)

if __name__ == "__main__":
    asyncio.run(chat())