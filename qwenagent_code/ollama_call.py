from qwen_agent.agents import Assistant
from qwen_agent.llm import get_chat_model

from qwenagent_api import read_steam_response

# 配置 DashScope API Key（需在环境变量或代码中设置）
import os
# os.environ['DASHSCOPE_API_KEY'] = 'your-api-key'

# 创建模型实例
llm = get_chat_model({
    'model': 'deepseek-r1:32b',
    'model_server': 'http://localhost:11434/v1',  # ⬅️ 必须是完整 OpenAI API 路径！
    'api_key': 'ollama'  # Ollama 要求非空，任意字符串即可
})

# 创建智能体
agent = Assistant(llm=llm)
response = agent.run([{'role': 'user', 'content': '你好！'}])
read_steam_response(response)