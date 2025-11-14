import ollama

# 调用模型生成回复，流式输出
for chunk in ollama.chat(
    model='deepseek-r1:32b',
    messages=[{'role': 'user', 'content': '讲一个笑话'}],
    stream=True
):
    print(chunk['message']['content'], end='', flush=True)