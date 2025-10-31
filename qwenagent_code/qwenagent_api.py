def read_steam_response(response_generator):
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

    return full_response