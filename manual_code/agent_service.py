import sys
import re
import json
from api_service import QueryService, SemanticService, AnalysisService

# 初始化API服务
queryService = QueryService()
semanticService = SemanticService()
analysisService = AnalysisService()

# metadata init  元数据初始向量化过程
def init():
    print("开始执行方法init")
    # 1. 从mysql中获取表结构元数据信息
    results = queryService.query(
        """
        SELECT t.TABLE_NAME AS '表名', t.TABLE_COMMENT AS '表备注', c.COLUMN_NAME AS '字段名', c.COLUMN_TYPE AS '字段类型', c.COLUMN_COMMENT AS '字段备注'
        FROM INFORMATION_SCHEMA.TABLES t
                 INNER JOIN
             INFORMATION_SCHEMA.COLUMNS c
             ON t.TABLE_NAME = c.TABLE_NAME AND t.TABLE_SCHEMA = c.TABLE_SCHEMA
        WHERE t.TABLE_SCHEMA = 'chaiys'
          and t.table_name in ('college_entrance_admission', 'college_entrance_examination')
        ORDER BY t.TABLE_NAME,
                 c.ORDINAL_POSITION
        """
    )
    table_data = {}
    for row in results:
        table_name = row[0]
        if table_name not in table_data:
            table_data[table_name] = {
                '表名': table_name,
                '表备注': row[1],
                '字段列表': []
            }

        # 添加字段信息
        table_data[table_name]['字段列表'].append({
            '字段名': row[2],
            '字段类型': row[3],
            '字段备注': row[4]
        })

    print(table_data)
    semanticService.create_index()
    # 转换为列表形式
    for table_name in list(table_data.keys()):
        table = table_data[table_name]
        # 2. 向量化表结构元数据信息并插入ES索引
        semanticService.vectorize_and_index(table_name, json.dumps(table, ensure_ascii=False))


# agent flow  智能体核心流程
def chat(user_query):
    print("执行方法chat")
    # 1. 语义匹配
    table = semanticService.hybrid_search(user_query, 1)
    if not table:
        print(f"未匹配到字段")
        return
    table_list = [t["table_info"] for t in table]
    prompt = f"""
        你是一个MySQL专家。根据以下表结构信息：
        {table_list}

        用户查询："{user_query}"

        生成标准MYSQL查询语句。
        要求：
        1. 只输出MYSQL语句，不要额外解释
        2. 根据语义和字段类型，使用COUNT/SUM/AVG等聚合函数进行计算，非必须
        3. 给生成的字段取一个简短的中文名称
        输出格式：使用[]包含sql文本即可，不需要其他输出，便于解析，例如:[select 1 from dual]
    """
    print(f"PROMPT={prompt}")
    # 2. 大模型生成SQL
    str1 = analysisService.analysis(prompt)
    sql = re.search(r'\[(.*?)\]', str1, re.DOTALL).group(1).strip()

    # 3. 执行查询
    if sql:
        resultSet = queryService.query_with_column(sql)
        print(f"resultSet={resultSet}")

        # 基础分析
        prompt2 = f"""
            根据以下表结构信息：
            {table_list}
            查询SQL：
            {sql}
            和以下数据信息：
            {resultSet}

            用户查询："{user_query}"

            生成一段简要分析，加上一些预测总结的内容
        """
        print(f"PROMPT={prompt2}")
        analysisService.analysis(prompt2)


if __name__ == "__main__":
    # 获取命令行参数
    args = sys.argv[1:]  # 第一个参数是脚本名，跳过

    if not args:
        print("请提供参数：init或者chat+user_query")
    elif args[0] == "init":
        init()
    elif args[0] == "chat":
        print(f"user_query={args[1]}")
        chat(args[1])
    else:
        print(f"未知参数: {args[0]}")