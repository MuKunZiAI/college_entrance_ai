import sys

from langchain.agents import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_ollama import OllamaLLM

from api_service import QueryService, SemanticService, AnalysisService

# 初始化服务
queryService = QueryService()
semanticService = SemanticService()
analysisService = AnalysisService()

# 初始化 Ollama
llm = OllamaLLM(
    model="qwen3:32b",
    callbacks=[StreamingStdOutCallbackHandler()]
)

# 自定义工具
def match_metadata(user_query: str) -> str:
    """语义匹配表结构，可多次调用"""
    table = semanticService.hybrid_search(user_query, 1)
    table_list = [t["table_info"] for t in table]
    return f"{table_list}"


def execute_sql(query: str) -> str:
    """执行SQL查询"""
    return queryService.query_with_column(query)


# 创建工具
semantic_tool = Tool(
    name="match_metadata",
    func=match_metadata,
    description="根据输入文本语义匹配表结构，每次返回一个最相关的表结构。对于需要多表查询的问题，需要多次调用此工具。"
)

sql_tool = Tool(
    name="execute_sql",
    func=execute_sql,
    description="执行SQL查询并返回结果。输入应为标准SQL语句。注意：可能需要执行多个SQL查询来获取不同表中的数据。"
)

# 初始化Agent
agent = initialize_agent(
    tools=[semantic_tool, sql_tool],
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=10,  # 增加最大迭代次数以支持多步查询
    early_stopping_method="generate"
)

def chat(user_query):
    print("执行方法chat")

    # 更明确的指令
    enhanced_query = f"""
    问题：{user_query}

    请特别注意：这个问题可能需要从多个表中查询数据。
    1. 首先确定需要查询哪些数据
    2. 使用match_metadata工具分别匹配包含这些数据的表结构
    3. 对每个表生成相应的SQL查询语句
    4. 执行查询并汇总结果
    5. 最后计算并给出答案

    请确保逐步执行，不要跳过任何步骤。
    """

    try:
        result = agent.run(enhanced_query)
        print(f"\n最终结果: {result}")
    except Exception as e:
        print(f"执行过程中出错: {str(e)}")
        # 这里可以添加重试或更详细的错误处理逻辑


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