import asyncio
import json
import re
from typing import Optional, TypedDict, List

from agentscope.agent import AgentBase
from agentscope.formatter import OllamaChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.message import Msg
from agentscope.model import OllamaChatModel
from agentscope.pipeline import sequential_pipeline
from api_service import QueryService, SemanticService

# 初始化 API 服务
queryService = QueryService()
semanticService = SemanticService()


# ========================
# 🧠 增强状态定义
# ========================
class State(TypedDict):
    original_intent: Optional[str]  # 如 "考生人数"
    known_table_struct: Optional[List]  # 已确认的表结构
    last_sql_template: Optional[str]  # 带占位符的 SQL 模板，如 "SELECT ... WHERE year = {year}"


state = State()


# 结果获取
async def get_response(result):
    responses = []
    async for item in result:
        for chunk in item.content:
            if chunk.get("type") == "thinking":
                print(chunk.get("thinking"))
            else:
                responses.append(chunk.get("text"))

    return "".join(responses).strip()

# json判断
def is_valid_json(json_str):
    """
    判断字符串是否为有效的JSON
    """
    try:
        json.loads(json_str)
        return True
    except (json.JSONDecodeError, ValueError):
        return False

# --- 模板匹配智能体 ---
class TemplateAgent(AgentBase):
    def __init__(self, name: str = "template_agent"):
        super().__init__()
        self.name = name
        self.sys_prompt = "你是一个参数提取器。"
        self.model = OllamaChatModel(
            model_name="qwen3:32b",
            stream=True,
            enable_thinking=True,
        )
        self.formatter = OllamaChatFormatter()
        self.memory = InMemoryMemory()

    async def reply(self, msg: Msg) -> Msg:
        """模板匹配尝试"""
        print("=== 🟦 Template Agent ===")
        user_query = msg.content
        history = await self.memory.get_memory()
        # ───────────────────────────────────────
        # 🔄 情况1：无上下文 → SQL Agent
        # ───────────────────────────────────────
        if (not state.get("known_table_struct")
                or not state.get("original_intent")
                or not len(history) >= 3
                or not state.get("last_sql_template")):
            print("⚠️ 不存在上下文，进入SQL Agent流程")
            return Msg(self.name, "NO_CONTEXT", role="assistant")

        # ───────────────────────────────────────
        # 🔄 情况2：已有上下文 → 尝试参数化追问处理
        # ───────────────────────────────────────
        print("🔄 检测到历史上下文，尝试参数化追问")
        try:
            # 1. 用 LLM 从追问中提取参数
            extract_prompt = f"""
                原始意图: {state['original_intent']}
                原始表结构：{state.get("known_table_struct")}
                当前问题: "{user_query}"

                请判断一下用户是否在进行年份的追问。
                1、如果不是对年份的追问，直接输出空JSON '{{}}'
                2、如果是追问，从中提取**变化的参数值**（年份），以 JSON 格式返回。
                示例：
                - “那2017年呢？” → {{"year": "2017"}}

                只输出 JSON，不要其他内容。
            """
            formatted = await self.formatter.format([
                Msg("system", self.sys_prompt, "system"),
                *history,
                Msg("user", extract_prompt, "user"),
            ])

            result = await self.model(formatted)
            json_str = await get_response(result)
            print(f"🔍 提取参数: {json_str}")
            params = json.loads(json_str)
            if not "year" in params:
                print("⚠️ 年份参数提取失败，进入改写流程")
                return Msg(self.name, "NO_MATCH", role="assistant")

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
                    content = {
                        "tableStruct": state.get("known_table_struct"),
                        "sql": new_sql,
                        "resultSet": result
                    }
                    return Msg(self.name, json.dumps(content, ensure_ascii=False), role="assistant")
            except KeyError as e:
                print(f"模板参数不匹配: {e}")
            except Exception as e:
                print(f"模板执行失败: {e}")

        except Exception as e:
            print(f"参数提取失败: {e}")

        print("⚠️ 追问处理失败，进入改写流程")
        return Msg(self.name, "NO_MATCH", role="assistant")

    async def observe(self, msg: Msg | list[Msg] | None) -> None:
        await self.memory.add(msg)


# --- 意图改写智能体 ---
class RewriteAgent(AgentBase):
    def __init__(self, name: str = "rewriter"):
        super().__init__()
        self.name = name
        self.sys_prompt = "你是一个查询意图理解与改写专家。"
        self.model = OllamaChatModel(
            model_name="qwen3:32b",
            stream=True,
            enable_thinking=True,
        )
        self.formatter = OllamaChatFormatter()
        self.memory = InMemoryMemory()

    async def reply(self, msg: Msg) -> Msg:
        """理解意图并改写"""
        print("=== 🟨 Rewrite Agent ===")
        user_query = msg.content

        # 从记忆中取出历史上下文
        history = await self.memory.get_memory()

        # 构造 prompt
        user_prompt = f"""
            用户的问题可能是不完整的追问或模糊描述，请结合上下文补全成一个清晰的问题。

            【当前问题】
            "{user_query}"

            请将上面的问题改写成一个**可独立理解、完整表达查询意图**的自然语言问题。
            要求：
            1. 根据原意图内容补充信息，完善条件即可，无须发散联想
            2. 不超过 50 字
            3. 仅输出改写后的问题内容
        """
        prompt = await self.formatter.format([
            Msg("system", self.sys_prompt, "system"),
            *history,
            Msg("user", user_prompt, "user"),
        ])

        result = await self.model(prompt)
        final_res = await get_response(result)
        print(f"✍️ 改写后问题：{final_res}")

        # 将改写结果返回给 SQL Agent
        return Msg(self.name, final_res, role="assistant")

    async def observe(self, msg: Msg | list[Msg] | None) -> None:
        await self.memory.add(msg)


# --- SQL Agent ---
class SQLAgent(AgentBase):
    def __init__(self, name: str = "sql_agent"):
        super().__init__()
        self.name = name
        self.sys_prompt = "你是一个 MySQL 专家。"
        self.model = OllamaChatModel(
            model_name="qwen3:32b",
            stream=True,
            enable_thinking=True,
        )
        self.formatter = OllamaChatFormatter()
        self.memory = InMemoryMemory()

    async def reply(self, msg: Msg) -> Msg:
        print("=== 🟦 Sql Agent ===")
        user_query = msg.content

        # 1. 语义匹配
        table = semanticService.hybrid_search(user_query, 1)
        if not table:
            return Msg(self.name, "未匹配到字段", role="assistant")
        table_struct = [t["table_info"] for t in table]

        # 2. prompt 构造
        user_prompt = f"""
            根据以下表结构信息：{table_struct}
            用户查询：" {user_query} "
            生成标准 MySQL 查询语句。要求：
            1. 只输出 MySQL 语句，不要额外解释
            2. 使用 COUNT / SUM / AVG 等聚合函数时注意字段类型
            3. 输出格式：[select ...]
        """

        prompt = await self.formatter.format([
            Msg("system", self.sys_prompt + user_prompt, "system"),
            *await self.memory.get_memory(),
        ])

        result = await self.model(prompt)
        final_res = await get_response(result)
        print(f"💡 大模型生成的SQL: {final_res}")
        m = re.search(r'\[(.*?)\]', final_res, re.DOTALL)
        if not m:
            return Msg(self.name, "SQL 生成失败", role="assistant")
        sql = m.group(1).strip()
        print(f"💡 最终SQL生成成功: {sql}")

        resultSet = queryService.query_with_column(sql)
        if not resultSet:
            return Msg(self.name, "SQL 查询失败", role="assistant")
        print(f"💡 数据查询成功: {resultSet}")

        # 生成 SQL 模板（简单年份/数字替换）
        template_sql = re.sub(r'\b(19|20)\d{{2}}\b', '{year}', sql)  # 年份
        template_sql = re.sub(r'\b\d+\b', '{year}', template_sql, count=1)  # 兜底替换第一个数字

        print(f"🧠 原始意图: {user_query}")
        print(f"📦 保存 SQL 模板: {template_sql}")

        state["original_intent"] = user_query
        state["known_table_struct"] = table_struct
        state["last_sql_template"] = template_sql

        content = {
            "tableStruct": table_struct,
            "sql": sql,
            "resultSet": resultSet
        }
        return Msg(self.name, json.dumps(content, ensure_ascii=False), role="assistant")

    async def observe(self, msg: Msg | list[Msg] | None) -> None:
        await self.memory.add(msg)


# --- 分析 Agent ---
class AnalysisAgent(AgentBase):
    def __init__(self, name: str = "analysis_agent"):
        super().__init__()
        self.name = name
        self.sys_prompt = "你是一个数据分析专家。"
        self.model = OllamaChatModel(
            model_name="qwen3:32b",
            stream=True,
            enable_thinking=True,
        )
        self.formatter = OllamaChatFormatter()
        self.memory = InMemoryMemory()

    async def reply(self, msg: Msg) -> Msg:
        print("=== 🟩 Analysis Agent ===")
        if not is_valid_json(msg.content):
            print(f"💥无有效数据进行分析：{msg.content}")
            return Msg(self.name, "无有效数据进行分析", role="assistant")
        data = json.loads(msg.content)
        if not isinstance(data, dict) or 'sql' not in data:
            print(f"💥无有效数据进行分析：{msg.content}")
            return Msg(self.name, "无有效数据进行分析", role="assistant")

        user_prompt = f"""
            根据以下表结构信息：{data['tableStruct']}
            查询 SQL：{data['sql']}
            以及数据信息：{data['resultSet']}
            生成一段简要回答。
        """
        prompt = await self.formatter.format([
            Msg("system", self.sys_prompt + user_prompt, "system"),
            *await self.memory.get_memory(),
        ])
        result = await self.model(prompt)
        final_res = await get_response(result)
        print(f"🧠 最终回答: {final_res}")
        return Msg(self.name, final_res, role="assistant")

    async def observe(self, msg: Msg | list[Msg] | None) -> None:
        await self.memory.add(msg)


# --- 聊天主流程 ---
async def chat():
    print("启动 AgentScope 多智能体系统（输入 exit 退出）")

    template_agent = TemplateAgent()
    rewrite_agent = RewriteAgent()
    sql_agent = SQLAgent()
    analysis_agent = AnalysisAgent()

    while True:
        user_input = await asyncio.get_event_loop().run_in_executor(
            None, input, "👤 用户: "
        )
        if user_input.strip().lower() == "exit":
            print("退出程序")
            break

        user_msg = Msg("user", user_input, role="user")
        # 所有 Agent 观察输入
        for agent in [template_agent, rewrite_agent, sql_agent, analysis_agent]:
            await agent.observe(user_msg)

        # Step 1️⃣ 模板匹配
        template_res = await template_agent.reply(user_msg)

        if template_res.content.strip().upper() == "NO_MATCH":
            # 模板失败 → 走 改写 → SQL → 分析
            print("🌀 模板匹配失败，进入改写流程")
            final_msg = await sequential_pipeline([rewrite_agent, sql_agent, analysis_agent], user_msg)
        elif template_res.content.strip().upper() == "NO_CONTEXT":
            # 无上下文 → 走 SQL → 分析
            print("🌀 无上下文，进入SQL流程")
            final_msg = await sequential_pipeline([sql_agent, analysis_agent], user_msg)
        else:
            # 模板成功匹配 → 走 SQL + 分析
            print("🎯 模板匹配成功，直接进入分析流程")
            final_msg = await sequential_pipeline([analysis_agent], template_res)

        # 所有 Agent 观察结果
        for agent in [template_agent, rewrite_agent, sql_agent, analysis_agent]:
            await agent.observe(final_msg)

        print("\n" + "-" * 50)


if __name__ == "__main__":
    asyncio.run(chat())