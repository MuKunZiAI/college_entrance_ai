import sys
import re
from langchain.chains import LLMChain, SequentialChain, TransformChain
from langchain.prompts import PromptTemplate
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
    callbacks=[StreamingStdOutCallbackHandler()],
)


# 第一步：语义匹配表结构
def get_table_schema(inputs: dict) -> dict:
    user_query = inputs["user_query"]
    table = semanticService.hybrid_search(user_query, 1)
    table_list = [t["table_info"] for t in table]
    return {"table_schema": table_list}

def find_sql(str):
    # 只提取SQL语句内容（不包括标记）
    pattern = r'```sql\s*([\s\S]*?)\s*;?\s*```'
    match = re.search(pattern, str, re.DOTALL)

    if match:
        sql_content = match.group(1).strip()
        return sql_content
    return ";"

# 第二步：生成并执行 SQL
def execute_sql(inputs: dict) -> dict:
    table_schema = inputs["table_schema"]
    user_query = inputs["user_query"]

    # 生成 SQL 的提示模板
    sql_prompt = PromptTemplate(
        input_variables=["table_schema", "user_query"],
        template="基于以下表结构：{table_schema}\n请根据用户查询生成SQL语句：{user_query}，markdown格式返回最终SQL"
    )
    sql_chain = LLMChain(llm=llm, prompt=sql_prompt)
    generated_sql = sql_chain.run({
        "table_schema": table_schema,
        "user_query": user_query
    })
    # 提取 SQL
    generated_sql = find_sql(generated_sql)
    # 执行 SQL
    sql_result = queryService.query_with_column(generated_sql)
    return {"sql_result": sql_result, "generated_sql": generated_sql}


# 第三步：分析结果
analysis_prompt = PromptTemplate(
    input_variables=["user_query", "sql_result", "generated_sql", "table_schema"],
    template="用户查询：{user_query}\n生成的SQL：{generated_sql}\n查询结果：{sql_result}\n请用中文分析结果："
)

analysis_chain = LLMChain(
    llm=llm,
    prompt=analysis_prompt,
    output_key="analysis_result"
)

# 构建顺序链
overall_chain = SequentialChain(
    chains=[
        TransformChain(
            input_variables=["user_query"],
            output_variables=["table_schema"],
            transform=get_table_schema
        ),
        TransformChain(
            input_variables=["user_query", "table_schema"],
            output_variables=["sql_result", "generated_sql"],
            transform=execute_sql
        ),
        analysis_chain
    ],
    input_variables=["user_query"],
    output_variables=["analysis_result"],
    verbose=True
)


# metadata init
def init():
    print("开始执行方法init")


# 新的顺序链调用
def chat(user_query):
    print("执行方法chat")
    result = overall_chain({"user_query": user_query})
    print(result["analysis_result"])


if __name__ == "__main__":
    args = sys.argv[1:]

    if not args:
        print("请提供参数：init或者chat+user_query")
    elif args[0] == "init":
        init()
    elif args[0] == "chat":
        print(f"user_query={args[1]}")
        chat(args[1])
    else:
        print(f"未知参数: {args[0]}")