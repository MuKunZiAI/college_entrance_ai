import asyncio
import re

from api_service import QueryService, SemanticService, AnalysisService, Result
from message_manage import MessageManager

# 初始化API服务
queryService = QueryService()
semanticService = SemanticService()
analysisService = AnalysisService()
# 初始化管理器，最多保留 10 轮对话（即 20 条消息）
mm = MessageManager(max_history=10)
session_id = "uuid1"

# sql agent
def sql_agent(user_query):
    # 1. 语义匹配
    table = semanticService.hybrid_search(user_query, 1)
    if not table:
        print(f"未匹配到字段")
        return Result.error()
    table_struct = [t["table_info"] for t in table]
    prompt = f"""
            你是一个MySQL专家。根据以下表结构信息：
            {table_struct}

            历史问答（仅供参考）："{mm.get_messages(session_id)}"
            
            用户查询："{user_query}"

            生成标准MYSQL查询语句。
            要求：
            1. 只输出MYSQL语句，不要额外解释
            2. 根据语义和字段类型，使用COUNT/SUM/AVG等聚合函数进行计算，非必须
            3. 给生成的字段取一个简短的中文名称
            输出格式：使用[]包含sql文本即可，不需要其他输出，便于解析，例如:[select 1 from dual]
        """
    print(f"SQL AGENT PROMPT={prompt}")
    # 2. 大模型生成SQL
    str1 = analysisService.analysis(prompt)
    sql = re.search(r'\[(.*?)\]', str1, re.DOTALL).group(1).strip()

    # 3. 执行查询
    if not sql:
        print("\nSQL生成失败")
        return Result.error()

    resultSet = queryService.query_with_column(sql)
    if not resultSet:
        print("\nSQL查询失败")
        return Result.error()
    return Result.success(data={ "tableStruct": table_struct, "resultSet": resultSet, "sql": sql })

# analysis agent
def analysis_agent(user_query, data):
    # 基础分析
    prompt = f"""
                根据以下表结构信息：
                {data['tableStruct']}
                
                查询SQL：
                {data['sql']}
                
                和以下数据信息：
                {data['resultSet']}

                历史问答（仅供参考）："{mm.get_messages(session_id)}"
                
                用户查询："{user_query}"

                生成一段简要分析，加上一些预测总结的内容
            """
    print(f"ANALYSIS AGENT PROMPT={prompt}")
    return Result.success(analysisService.analysis(prompt))

def workflow(user_input):
    # 1 - SQL Agent
    result = sql_agent(user_input)
    if not result.success:
        return None
    # 2 - Analysis Agent
    return analysis_agent(user_input, result.data)

# agent flow  智能体核心流程
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
        # 执行工作流
        response = workflow(user_input)
        # 缓存历史对话
        if response and response.success:
            mm.add_user_message(session_id, user_input)
            mm.add_assistant_message(session_id, response.data)

        print("\n")
        print("-" * 50)

if __name__ == "__main__":
    asyncio.run(chat())