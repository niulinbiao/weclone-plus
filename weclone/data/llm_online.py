import requests

def api_infer(inputs: List[str], model_name: str, **kwargs) -> List[str]:
    """
    用API调用大模型，返回结果列表，每个对应输入的返回文本。
    """
    results = []
    api_url = "https://api.your-llm-provider.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {your_api_key}",
        "Content-Type": "application/json"
    }

    for prompt in inputs:
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            # 其他参数根据API要求填写，如temperature, max_tokens等
            "temperature": kwargs.get("temperature", 0),
        }
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        # 根据API响应格式提取文本
        text = data["choices"][0]["message"]["content"]
        results.append(text)

    return results
