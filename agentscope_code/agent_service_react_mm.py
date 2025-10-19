import asyncio
import re

from agentscope.agent import AgentBase  # 或者 agentscope.agent.Agent
from agentscope.formatter import OllamaChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.message import Msg
from agentscope.model import OllamaChatModel
from agentscope.pipeline import MsgHub, sequential_pipeline
from api_service import QueryService, SemanticService

# 初始化 API 服务
queryService = QueryService()
semanticService = SemanticService()

# --- 定义 SQL Agent ---
class SQLAgent(AgentBase):
    def __init__(self, name: str = "sql_agent"):
        super().__init__()
        self.name = name
        self.sys_prompt = "你是一个 MySQL 专家。"
        self.model = OllamaChatModel(
            model_name="qwen3:32b",  # 指定模型名称
            stream=True,  # 根据需要设置是否流式输出
            enable_thinking=True,  # 为Qwen3启用思考功能（可选）
            # host="http://localhost:11434" # 如果Ollama不在默认地址，需指定
        )
        self.formatter = OllamaChatFormatter()
        self.memory = InMemoryMemory()

    async def reply(self, msg: Msg) -> Msg:
        """直接调用大模型，产生回复消息。"""
        user_query = msg.content  # Msg 的 content 是用户的文本
        await self.memory.add(msg)
        # 1. 语义匹配
        table = semanticService.hybrid_search(user_query, 1)
        if not table:
            return Msg(self.name, "未匹配到字段", role="assistant")
        table_struct = [t["table_info"] for t in table]
        # 构造 prompt 给 SQL 模型服务
        user_prompt = f"""
            根据以下表结构信息：{table_struct}

            用户查询：" {user_query} "

            生成标准 MySQL 查询语句。要求：
            1. 只输出 MySQL 语句，不要额外解释
            2. 根据语义与字段类型，可能使用 COUNT / SUM / AVG 等聚合函数
            3. 给生成的字段取简短中文名称
            输出格式：使用 [ 和 ] 包括 sql 文本即可，例如：[select 1 from dual]
        """
        prompt = await self.formatter.format(
            [
                Msg("system", self.sys_prompt + user_prompt, "system"),
                *await self.memory.get_memory(),
            ],
        )
        # 调用模型
        result = await self.model(prompt)
        responses = []
        async for item in result:
             for chunk in item.content:
                if chunk.get("type") == "thinking":
                    print(chunk.get("thinking"))
                else:
                    responses.append(chunk.get("text"))
        final_res = "".join(responses)
        # 返回
        str1 = final_res
        m = re.search(r'\[(.*?)\]', str1, re.DOTALL)
        if not m:
            return Msg(self.name, "SQL 生成失败", role="assistant")
        sql = m.group(1).strip()

        # 执行 SQL
        resultSet = queryService.query_with_column(sql)
        if not resultSet:
            return Msg(self.name, "SQL 查询失败", role="assistant")

        # 用 Msg 携带数据给下游 agent
        # content 可以是一个结构化字符串 / JSON，也可以封装成 Msg 的 metadata
        content = {
            "tableStruct": table_struct,
            "sql": sql,
            "resultSet": resultSet
        }
        # 在记忆中记录响应
        msg = Msg(self.name, content, role="assistant")
        await self.memory.add(msg)

        return msg

    async def observe(self, msg: Msg | list[Msg] | None) -> None:
        """观察消息。"""
        # 将消息存储在记忆中
        await self.memory.add(msg)

    async def handle_interrupt(self) -> Msg:
        """处理中断。"""
        # 以固定响应为例
        return Msg(
            name=self.name,
            content="我注意到您打断了我的回复，我能为你做些什么？",
            role="assistant",
        )

# --- 定义 Analysis Agent ---
class AnalysisAgent(AgentBase):
    def __init__(self, name: str = "analysis_agent"):
        super().__init__()
        self.name = name
        self.sys_prompt = "你是一个 数据分析 专家。"
        self.model = OllamaChatModel(
            model_name="qwen3:32b",  # 指定模型名称
            stream=True,  # 根据需要设置是否流式输出
            enable_thinking=True,  # 为Qwen3启用思考功能（可选）
            # host="http://localhost:11434" # 如果Ollama不在默认地址，需指定
        )
        self.formatter = OllamaChatFormatter()
        self.memory = InMemoryMemory()

    async def reply(self, msg: Msg) -> Msg:
        # msg.content 是上游 SQLAgent 输出的结构数据
        data = msg.content
        # 为安全起见，可以校验 data 是否有字段
        if not isinstance(data, dict) or 'sql' not in data:
            return Msg(self.name, "无有效数据进行分析", role="assistant")

        # 生成 prompt 做分析
        user_prompt = f"""
            根据以下表结构信息：{data['tableStruct']}
            查询 SQL：{data['sql']}
            以及数据信息：{data['resultSet']}
                        
            用户查询："{msg.content}"  # 注意：Msg.origin 应指用户的 Msg
            生成一段简要分析，并做预测总结
        """
        prompt = await self.formatter.format(
            [
                Msg("system", self.sys_prompt + user_prompt, "system"),
                *await self.memory.get_memory(),
            ],
        )
        # 调用模型
        result = await self.model(prompt)
        responses = []
        async for item in result:
             for chunk in item.content:
                if chunk.get("type") == "thinking":
                    print(chunk.get("thinking"))
                else:
                    responses.append(chunk.get("text"))
        final_res = "".join(responses)
        return Msg(self.name, final_res, role="assistant")

    async def observe(self, msg: Msg | list[Msg] | None) -> None:
        """观察消息。"""
        # 将消息存储在记忆中
        await self.memory.add(msg)

    async def handle_interrupt(self) -> Msg:
        """处理中断。"""
        # 以固定响应为例
        return Msg(
            name=self.name,
            content="我注意到您打断了我的回复，我能为你做些什么？",
            role="assistant",
        )

# --- 聊天主流程 ---
async def chat():
    print("启动 AgentScope 多智能体系统（输入 exit 退出）")

    # 同步消息 hub（如果你要多个 agent 并发/广播可用 MsgHub）
    while True:
        user_input = await asyncio.get_event_loop().run_in_executor(
            None, input, "用户: "
        )
        if user_input.strip().lower() == 'exit':
            print("退出程序")
            break
        if not user_input.strip():
            continue

        # 构造用户消息
        msg = Msg("user", user_input, role="user")
        sql_agent = SQLAgent()
        analysis_agent = AnalysisAgent()
        agents = [sql_agent, analysis_agent]

        # pipeline 的输入为用户的 msg
        # sequential_pipeline 返回最终的 Msg（最后 agent 的 reply 结果）
        final_msg = await sequential_pipeline(agents, msg)
        print(final_msg)
        print("\n" + "-" * 50)


if __name__ == "__main__":
    asyncio.run(chat())