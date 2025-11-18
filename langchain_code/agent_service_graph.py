import re
import json
from typing import Annotated, Literal, Optional, List
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict
from pydantic import BaseModel, Field  # 正确导入 Pydantic v2

# 这些服务请参考前几篇文章
from api_service import QueryService, SemanticService, AnalysisService

# 初始化服务
queryService = QueryService()
semanticService = SemanticService()
analysisService = AnalysisService()

# 初始化 LLM
llm = ChatOllama(model="qwen3:32b", temperature=0.3)


# ========================
# 🛠️ 工具：结构化 SQL 生成
# ========================
class GenerateSQL(BaseModel):
    """生成一个安全的 SQL 查询语句"""
    sql: str = Field(description="标准的 MySQL 查询语句，仅包含 SELECT")

llm_with_tool = llm.bind_tools([GenerateSQL], tool_choice="GenerateSQL")


# ========================
# 🧠 增强状态定义
# ========================
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    original_intent: Optional[str]  # 如 "考生人数"
    known_table_struct: Optional[List]  # 已确认的表结构
    last_sql_template: Optional[str]  # 带占位符的 SQL 模板，如 "SELECT ... WHERE year = {year}"
    query_result: Optional[list]  # 最近一次查询结果
    next_agent: str

# ========================
# 移除think内容
# ========================
def remove_think_content(text):
    """
    移除<think>标签及其所有内容，支持多行
    """
    # 使用re.DOTALL标志
    cleaned = re.sub(r'<think>.*?</think>', '', text.strip(), flags=re.DOTALL)

    return cleaned.strip()

# ========================
# 🍻模板匹配 智能体（支持追问模板复用）
# ========================
def template_agent(state: State):
    print("=== 🟦 Template Agent ===")
    current_query = state["messages"][-1].content

    # ───────────────────────────────────────
    # 🔄 情况1：无上下文 → SQL Agent
    # ───────────────────────────────────────
    if (not state.get("known_table_struct")
            or not state.get("original_intent")
            or not state.get("last_sql_template")):
        print("⚠️ 不存在上下文，进入SQL Agent流程")
        return {
            "messages": state["messages"],
            "next_agent": "sql"
        }

    # ───────────────────────────────────────
    # 🔄 情况2：已有上下文 → 尝试参数化追问处理
    # ───────────────────────────────────────
    print("🔄 检测到历史上下文，尝试参数化追问")
    try:
        # 1. 用 LLM 从追问中提取参数
        extract_prompt = f"""
            你是一个参数提取器。

            原始意图: {state['original_intent']}
            当前问题: "{current_query}"

            请判断一下用户是否在进行年份的追问。
            1、如果不是对年份的追问，直接输出空JSON '{{}}'
            2、如果是追问，从中提取**变化的参数值**（年份），以 JSON 格式返回。
            示例：
            - “那2017年呢？” → {{"year": "2017"}}

            只输出 JSON，不要其他内容。
        """
        param_resp = llm.invoke([
            SystemMessage(content=extract_prompt),
            HumanMessage(content=current_query)
        ])
        json_str = remove_think_content(param_resp.content)
        print(f"🔍 提取参数: {json_str}")
        params = json.loads(json_str)
        if not "year" in params:
            print("⚠️ 年份参数提取失败，进入改写流程")
            return {
                "messages": state["messages"],
                "next_agent": "rewrite"
            }

        # 2. 填充 SQL 模板
        try:
            new_sql = state["last_sql_template"].format(**params)
            print(f"🛠️ 生成新 SQL: {new_sql}")
            # 3. 安全校验
            if not new_sql.strip().upper().startswith("SELECT"):
                raise ValueError("非 SELECT 语句")
            # 4. 执行查询
            result = queryService.query_with_column(new_sql)
            if result is not None:
                print(f"✅ 追问查询成功，返回 {len(result)} 行")
                return {
                    "query_result": result,
                    "next_agent": "analysis"
                }
        except KeyError as e:
            print(f"模板参数不匹配: {e}")
        except Exception as e:
            print(f"模板执行失败: {e}")

    except Exception as e:
        print(f"参数提取失败: {e}")

    print("⚠️ 追问处理失败，进入改写流程")
    return {
        "messages": state["messages"],
        "next_agent": "rewrite"
    }

# ========================
# 📌 改写智能体（语义补全 / 追问修正）
# ========================
def rewrite_agent(state: State):
    print("=== 🟨 Rewrite Agent ===")
    current_query = state["messages"][-1].content
    history = state["messages"]

    rewrite_prompt = f"""
        你是一个问题改写专家。
        用户的问题可能是不完整的追问或模糊描述，请结合上下文补全成一个清晰的问题。

        【当前问题】
        "{current_query}"

        请将上面的问题改写成一个**可独立理解、完整表达查询意图**的自然语言问题。
        要求：
        1. 根据原意图补充信息，完善条件
        2. 不超过 50 字
        3. 仅输出改写后的问题内容
    """

    response = llm.invoke([
        SystemMessage(content=rewrite_prompt),
        *history,
        HumanMessage(content=current_query)
    ])
    rewritten = remove_think_content(response.content)
    print(f"✍️ 改写后问题: {rewritten}")

    # 返回改写后的问题，让下一个 SQL Agent 直接使用
    return {
        "messages": [HumanMessage(content=rewritten)],
        "next_agent": "sql"
    }

# ========================
# 🕵️ SQL 智能体（支持记忆复用）
# ========================
def sql_agent(state: State):
    print("=== 🟦 Sql Agent ===")
    current_query = state["messages"][-1].content
    print("🆕 启动完整查询流程")

    # 1. 语义搜索表结构
    table_matches = semanticService.hybrid_search(current_query, 1)
    if not table_matches:
        return {
            "messages": [AIMessage(content="❌ 未找到相关数据表，请尝试更具体的问题。")],
            "next_agent": "end"
        }

    table_struct = [t["table_info"] for t in table_matches]
    print(f"📚 匹配表结构: {table_struct}")

    # 2. 生成 SQL
    sql_prompt = f"""
        你是一个 MySQL 专家，请根据以下表结构生成**只读查询（SELECT）**：

        表结构:
        {table_struct}

        用户问题: "{current_query}"

        要求:
        - 仅使用 SELECT，禁止写操作
        - 使用 COUNT/SUM/AVG 等聚合函数（如适用）
        - 给结果字段起简短中文别名（如 AS '考生人数'）
        - 不要输出解释，只通过工具返回 SQL 和说明
    """
    messages = [SystemMessage(content=sql_prompt), HumanMessage(content=current_query)]
    response = llm_with_tool.invoke(messages)

    # 3. 提取 SQL
    sql = None
    try:
        tool_calls = response.tool_calls
        if tool_calls:
            tool_call = tool_calls[0]
            sql = tool_call["args"]["sql"]
        else:
            raise ValueError("无法提取 SQL")
    except Exception as e:
        return {
            "messages": [AIMessage(content=f"💥 SQL 生成失败: {e}")],
            "next_agent": "end"
        }

    # 4. 执行查询
    result = queryService.query_with_column(sql)
    if result is None or len(result) == 0:
        print("🔍 查询无结果。")
        return {
            "messages": [AIMessage(content="🔍 查询无结果。")],
            "next_agent": "end"
        }
    print(f"🔍 查询成功，结果：{result}")

    # 5. 生成 SQL 模板（简单年份/数字替换）
    template_sql = re.sub(r'\b(19|20)\d{{2}}\b', '{year}', sql)  # 年份
    template_sql = re.sub(r'\b\d+\b', '{year}', template_sql, count=1)  # 兜底替换第一个数字

    print(f"🧠 原始意图: {current_query}")
    print(f"📦 保存 SQL 模板: {template_sql}")

    return {
        "original_intent": current_query,
        "known_table_struct": table_struct,
        "last_sql_template": template_sql,
        "query_result": result,
        "next_agent": "analysis"
    }


# ========================
# 📊 分析智能体
# ========================
def analysis_agent(state: State):
    print("=== 🟩 Analysis Agent ===")

    original_intent = state.get("original_intent") or "数据分析"
    table_struct = state.get("known_table_struct") or "未知"
    result = state["query_result"]

    analysis_prompt = f"""
        你是一名数据分析师，请基于以下信息生成专业、简洁的中文分析报告：

        - **用户意图**: {original_intent}
        - **表结构**: {table_struct}
        - **查询结果**: {result}

        要求:
        1. 先陈述事实（如“2017年考生人数为 12,345 人”）
        2. 再提供简要洞察或趋势预测
        3. 若数据单一，避免过度解读
        4. 语气友好、专业
    """

    response = llm.invoke([SystemMessage(content=analysis_prompt)])
    return {
        "messages": [response],
        "next_agent": "end"
    }


# ========================
# 🧭 路由函数
# ========================
def route_to_next_agent(state: State) -> Literal["sql", "analysis", "rewrite", "__end__"]:
    if state.get("next_agent") == "sql":
        return "sql"
    if state.get("next_agent") == "analysis":
        return "analysis"
    if state.get("next_agent") == "rewrite":
        return "rewrite"
    return "__end__"


# ========================
# 🌐 构建工作流
# ========================
workflow = StateGraph(State)

workflow.add_node("template", template_agent)
workflow.add_node("rewrite", rewrite_agent)   # 新增改写智能体
workflow.add_node("sql", sql_agent)
workflow.add_node("analysis", analysis_agent)

workflow.add_edge(START, "template")

# Template 之后根据条件流向
workflow.add_conditional_edges(
    "template",
    route_to_next_agent,
    {
        "rewrite": "rewrite",
        "sql": "sql",
        "analysis": "analysis",
        "__end__": END
    }
)

# 改写智能体完成后交给 SQL
workflow.add_edge("rewrite", "sql")

# SQL 执行后流向分析
workflow.add_edge("sql", "analysis")
workflow.add_edge("analysis", END)

# 启用记忆（支持多轮对话）
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# ========================
# ▶️ 测试运行
# ========================
if __name__ == "__main__":
    from uuid import uuid4

    print("🚀 智能数据助手（支持追问、多轮上下文）")
    print("=" * 80)
    print("💡 输入问题开始对话，输入 'exit' 退出。")
    print("=" * 80)

    # 每次启动生成唯一 thread_id（上下文记忆关键）
    thread_id = str(uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    while True:
        user_input = input("\n👤 你: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("👋 再见！会话已结束。")
            break

        # 构造输入
        input_data = {
            "messages": [HumanMessage(content=user_input)],
        }

        try:
            # 流式执行
            for chunk in app.stream(input_data, config=config):
                if "analysis" in chunk:
                    msg = chunk["analysis"]["messages"][0]
                    print(f"💬 助手: {msg.content.strip()}")
                    print("-" * 60)
                elif "sql" in chunk and "messages" in chunk["sql"]:
                    # SQL 生成失败的情况
                    for m in chunk["sql"]["messages"]:
                        if isinstance(m, AIMessage):
                            print(f"💬 助手: {m.content.strip()}")
                            print("-" * 60)
                elif "__end__" in chunk:
                    print("✅ 对话结束")

        except json.JSONDecodeError:
            print("⚠️ JSON 解析失败，可能是 LLM 返回了非结构化内容。")
        except KeyboardInterrupt:
            print("\n⏹️ 中断。输入 exit 可退出。")
        except Exception as e:
            print(f"💥 出错：{e}")