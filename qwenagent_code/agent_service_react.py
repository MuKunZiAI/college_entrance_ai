import sys
import json
from qwen_agent.agents import Assistant
from qwen_agent.tools import BaseTool
from api_service import QueryService, SemanticService, AnalysisService

# 初始化服务
queryService = QueryService()
semanticService = SemanticService()
analysisService = AnalysisService()


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


def chat(user_query):
    print("执行方法chat")
    try:
        # 运行Agent
        messages = [{'role': 'user', 'content': user_query}]
        response_generator = agent.run(messages=messages)
        # 处理生成器响应
        full_response = ''
        start = 0
        end = 0
        for response in response_generator:
            # 检查响应类型并适当处理
            if isinstance(response, list):
                # 如果是列表，提取内容
                for item in response:
                    if isinstance(item, dict) and 'content' in item:
                        full_response = item['content']
                        end = full_response.__len__()
                    elif isinstance(item, str):
                        full_response = item
                        end = full_response.__len__()
            elif isinstance(response, dict) and 'content' in response:
                full_response = response['content']
                end = full_response.__len__()
            elif isinstance(response, str):
                full_response = response
                end = full_response.__len__()
            print(f"{full_response[start:end]}", end="")
            start = end

        print(f"最终结果: {full_response}")
        return full_response
    except Exception as e:
        print(f"执行过程中出错: {str(e)}")
        # 这里可以添加重试或更详细的错误处理逻辑
        return f"错误: {str(e)}"


if __name__ == "__main__":
    args = sys.argv[1:]

    if not args:
        print("请提供参数：init或者chat+user_query")
    elif args[0] == "init":
        print("开始执行方法init")
        # 这里可以添加初始化逻辑
    elif args[0] == "chat":
        print(f"user_query={args[1]}")
        chat(args[1])
    else:
        print(f"未知参数: {args[0]}")